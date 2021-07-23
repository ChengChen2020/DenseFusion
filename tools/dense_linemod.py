import warnings
warnings.filterwarnings("ignore")

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
from lib.utils import setup_logger

import pytorch_ssim
import pytorch_msssim

from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader,
)
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

from tqdm import tqdm

from torchvision.utils import save_image

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(device)

# Initialize a perspective camera.
cameras = PerspectiveCameras(
    device=device,
    focal_length=((-572.4114, -573.57043),),
    principal_point=((325.2611, 242.04899),),
    image_size=((640, 480),),
)

raster_settings = RasterizationSettings(
    image_size=(480, 640),
    blur_radius=0.0,
    faces_per_pixel=1,
)

phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, cameras=cameras)
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = './datasets/linemod/Linemod_preprocessed', help='dataset root dir')
parser.add_argument('--model', type=str, default = './trained_models/linemod/pose_model_9_0.01310166542980859.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = './trained_models/linemod/pose_refine_model_493_0.006761023565178073.pth',  help='resume PoseRefineNet model')
opt = parser.parse_args()

opt.outf = './trained_models/linemod'
opt.log_dir = './experiments/logs/linemod'

num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points = 500
iteration = 4
bs = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir = 'experiments/eval_result/linemod'
knn = KNearestNeighbor(1)

meshes = []
for item in objlist:
    verts, faces = load_ply("{0}/models/obj_{1}.ply".format(opt.dataset_root, '%02d' % item))
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    meshes.append(mesh)

estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(opt.model))
refiner.load_state_dict(torch.load(opt.refine_model))

optimizer = torch.optim.Adam(refiner.parameters(), lr=1e-4)

dataset = PoseDataset_linemod('train', num_points, True, opt.dataset_root, 0.03, True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=10)
testdataset = PoseDataset_linemod('test', num_points, False, opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

print(len(dataloader))
d = iter(dataloader).next()
for i in d:
    print(i.shape)
print(d[-1])

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.safe_load(meta_file)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
#print(diameter)

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/diff_result_logs.txt'.format(output_result_dir), 'w')

mse_loss = nn.MSELoss()
ssim_loss = pytorch_ssim.SSIM()
msssim_loss = pytorch_msssim.MSSSIM()

best_test = np.Inf

st_time = time.time()

for epoch in range(1):
    estimator.eval()
    refiner.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloader)
    for i, data in enumerate(train_bar, 0):

        points, choose, img, target, model_points, label, idx = data
        #print(img.shape)
        if len(points.size()) == 2:
            print('No.{0} NOT Pass! Lost detection!'.format(i))
            fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
            continue
        points, choose, img, target, model_points, label, idx = Variable(points).cuda(), \
                                                                Variable(choose).cuda(), \
                                                                Variable(img).cuda(), \
                                                                Variable(target).cuda(), \
                                                                Variable(model_points).cuda(), \
                                                                Variable(label, requires_grad=False).cuda(), \
                                                                Variable(idx).cuda()

        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1)
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1)
        my_pred = torch.cat([my_r, my_t])

        #print('RT:', my_r, my_t)

        for ite in range(0, 1):

            T = my_t.view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)

            my_mat = torch.ones(4).cuda()
            my_mat = torch.diag(my_mat)
            my_mat[:3, :3] = quaternion_to_matrix(my_r)

            R = my_mat[:3, :3].view(1, 3, 3)
            my_mat[0:3, 3] = my_t

            #print('mat', my_mat)

            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        
            my_r_2 = pred_r.view(-1)
            my_t_2 = pred_t.view(-1)

            my_mat_2 = torch.ones(4).cuda()
            my_mat_2 = torch.diag(my_mat_2)
            my_mat_2[:3, :3] = quaternion_to_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2
        
            #print('mat2', my_mat_2)

            my_mat_final = torch.mm(my_mat, my_mat_2)

            #print('mat final', my_mat_final)

            my_r_final = my_mat_final.clone()
            my_r_final[0:3, 3] = 0
            my_r_final = matrix_to_quaternion(my_r_final[:3, :3])

            my_t_final = my_mat_final[0:3, 3]
            my_pred = torch.cat([my_r_final, my_t_final])

            my_r = my_r_final
            my_t = my_t_final

            # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

        my_r = quaternion_to_matrix(my_r)

        image_ref = phong_renderer(
            device=device,
            meshes_world=meshes[idx[0].item()],
            R=(my_r).T.unsqueeze(0),
            T=(my_t * 1000.).unsqueeze(0)
        )

        mm, _ = image_ref[..., :3].max(-1)
        #image_binary = (mm != 1).type(torch.float32)
        image_binary = torch.ones_like(mm) - mm

        #print(idx[0].item())
        #print(image_binary.dtype)
        #print(label.dtype)

        #save_image(image_binary, 'binary.png')
        #save_image(label[..., 0].type(torch.float32), 'target.png')

        #print(image_binary.shape)
        #print(torch.sum(image_binary))
        #print(torch.sum((image_binary > 0.).type(torch.int32)))
        #print(label.shape)
        #print(torch.sum(label))
        #print(torch.sum((image_binary - label) ** 2))

        #loss = torch.sum((image_binary - label[..., 0]) ** 2)
        loss = mse_loss(image_binary, label[..., 0])
        #loss = -ssim_loss(image_binary.unsqueeze(0), label[..., 0].unsqueeze(0).type(torch.float32))
        #loss = -msssim_loss(image_binary.unsqueeze(0), label[..., 0].unsqueeze(0).type(torch.float32))
        loss.backward()
            
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_num += points.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, 1, total_loss / total_num))
        
    print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

    logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
    logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
    test_dis = 0.0
    test_count = 0
    estimator.eval()
    refiner.eval()

    for j, data in enumerate(testdataloader, 0):
        points, choose, img, target, model_points, label, idx = data
        points, choose, img, target, model_points, label, idx = Variable(points).cuda(), \
                                                                Variable(choose).cuda(), \
                                                                Variable(img).cuda(), \
                                                                Variable(target).cuda(), \
                                                                Variable(model_points).cuda(), \
                                                                Variable(label).cuda(), \
                                                                Variable(idx).cuda()
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, 0.015, True)

        for ite in range(0, iteration):
            pred_r, pred_t = refiner(new_points, emb, idx)
            dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

        test_dis += dis.item()
        logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis.item()))

        test_count += 1

    test_dis = test_dis / test_count
    logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
    if test_dis <= best_test:
        best_test = test_dis
        torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
        print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
