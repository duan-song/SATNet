import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2

from models.SATNet import SATNet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='1', help='select gpu id')
parser.add_argument('--test_path',type=str,default='../',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
elif opt.gpu_id=='2':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print('USE GPU 2')
elif opt.gpu_id=='3':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print('USE GPU 3')
#load the model
model = CPNet()
pth_name = './cpts/epoch_best.pth'
print(pth_name)
model.load_state_dict(torch.load(pth_name), strict=False)
model.cuda()
model.eval()

test_datasets = ['DUT','SSD','RGBD135','LFSD','NJU2K','NLPR','SIP','STERE']
#test_datasets = ['LFSD','NJU2K','NLPR']
#test_datasets = ['STEREO']
for dataset in test_datasets:
    save_path = './test_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth_pseudo_DAM/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth = depth.repeat(1,3,1,1).cuda()
        res, res2, res3 = model(image,depth)
        pre = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        pre = pre.sigmoid().data.cpu().numpy().squeeze()
        pre = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)

        print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, pre*255)
        
    print('Test Done!')
   
