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
pth_name = './cpts/STMC_epoch_best.pth'
print(pth_name)
model.load_state_dict(torch.load(pth_name), strict=False)
model.cuda()
model.eval()

test_datasets = ['DUT','SSD','RGBD135','LFSD','NJU2K','NLPR','SIP','STERE']
#test_datasets = ['LFSD','NJU2K','NLPR']
#test_datasets = ['STEREO']
for dataset in test_datasets:
    save_path = './test_maps/ablation_r1/train_dam_test_dam2/' + dataset + '/'
#    before_save_path = './test_maps/Attention_visualization/Before_CIPN/'
#    after_save_path = './test_maps/Attention_visualization/After_CIPN/'
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

#        # save edge
#        save_edge_root = os.path.join(save_path, 'EdgeMap')
#        if not os.path.exists(save_edge_root):
#            os.makedirs(save_edge_root)
#        res2 = F.upsample(res2, size=gt.shape, mode='bilinear', align_corners=False)
#        res2 = res2.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_edge_root, name), res2 * 255)
#
#        # save semantic
#        save_semantic_root = os.path.join(save_path, 'SemanticMap')
#        if not os.path.exists(save_semantic_root):
#            os.makedirs(save_semantic_root)
#        res3 = F.upsample(res3, size=gt.shape, mode='bilinear', align_corners=False)
#        res3 = res3.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_semantic_root, name), res3 * 255)
#
#        # save layer 1 edge, semantic, fusion feature map
#        save_Layer_1_edge_root = os.path.join(save_path, 'Layer_1_edge')
#        if not os.path.exists(save_Layer_1_edge_root):
#            os.makedirs(save_Layer_1_edge_root)
#        layer_1_edge = F.upsample(edge_list[0], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_1_edge = layer_1_edge.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_1_edge_root, name), layer_1_edge * 255)
#
#        save_Layer_1_semantic_root = os.path.join(save_path, 'Layer_1_semantic')
#        if not os.path.exists(save_Layer_1_semantic_root):
#            os.makedirs(save_Layer_1_semantic_root)
#        layer_1_semantic = F.upsample(sal_list[0], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_1_semantic = layer_1_semantic.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_1_semantic_root, name), layer_1_semantic * 255)
#
#        save_Layer_1_fusion_root = os.path.join(save_path, 'Layer_1_fusion')
#        if not os.path.exists(save_Layer_1_fusion_root):
#            os.makedirs(save_Layer_1_fusion_root)
#        layer_1_fusion = F.upsample(fusion_list[0], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_1_fusion = layer_1_fusion.data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_1_fusion_root, name), layer_1_fusion * 255)
#
#        # save layer 2 edge, semantic, fusion feature map
#        save_Layer_2_edge_root = os.path.join(save_path, 'Layer_2_edge')
#        if not os.path.exists(save_Layer_2_edge_root):
#            os.makedirs(save_Layer_2_edge_root)
#        layer_2_edge = F.upsample(edge_list[1], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_2_edge = layer_2_edge.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_2_edge_root, name), layer_2_edge * 255)
#
#        save_Layer_2_semantic_root = os.path.join(save_path, 'Layer_2_semantic')
#        if not os.path.exists(save_Layer_2_semantic_root):
#            os.makedirs(save_Layer_2_semantic_root)
#        layer_2_semantic = F.upsample(sal_list[1], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_2_semantic = layer_2_semantic.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_2_semantic_root, name), layer_2_semantic * 255)
#
#        save_Layer_2_fusion_root = os.path.join(save_path, 'Layer_2_fusion')
#        if not os.path.exists(save_Layer_2_fusion_root):
#            os.makedirs(save_Layer_2_fusion_root)
#        layer_2_fusion = F.upsample(fusion_list[1], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_2_fusion = layer_2_fusion.data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_2_fusion_root, name), layer_2_fusion * 255)
#
#        # save layer 3 edge, semantic, fusion feature map
#        save_Layer_3_edge_root = os.path.join(save_path, 'Layer_3_edge')
#        if not os.path.exists(save_Layer_3_edge_root):
#            os.makedirs(save_Layer_3_edge_root)
#        layer_3_edge = F.upsample(edge_list[2], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_3_edge = layer_3_edge.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_3_edge_root, name), layer_3_edge * 255)
#
#        save_Layer_3_semantic_root = os.path.join(save_path, 'Layer_3_semantic')
#        if not os.path.exists(save_Layer_3_semantic_root):
#            os.makedirs(save_Layer_3_semantic_root)
#        layer_3_semantic = F.upsample(sal_list[2], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_3_semantic = layer_3_semantic.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_3_semantic_root, name), layer_3_semantic * 255)
#
#        save_Layer_3_fusion_root = os.path.join(save_path, 'Layer_3_fusion')
#        if not os.path.exists(save_Layer_3_fusion_root):
#            os.makedirs(save_Layer_3_fusion_root)
#        layer_3_fusion = F.upsample(fusion_list[2], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_3_fusion = layer_3_fusion.data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_3_fusion_root, name), layer_3_fusion * 255)
#
#        # save layer 4 edge, semantic, fusion feature map
#        save_Layer_4_edge_root = os.path.join(save_path, 'Layer_4_edge')
#        if not os.path.exists(save_Layer_4_edge_root):
#            os.makedirs(save_Layer_4_edge_root)
#        layer_4_edge = F.upsample(edge_list[3], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_4_edge = layer_4_edge.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_4_edge_root, name), layer_4_edge * 255)
#
#        save_Layer_4_semantic_root = os.path.join(save_path, 'Layer_4_semantic')
#        if not os.path.exists(save_Layer_4_semantic_root):
#            os.makedirs(save_Layer_4_semantic_root)
#        layer_4_semantic = F.upsample(sal_list[3], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_4_semantic = layer_4_semantic.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_4_semantic_root, name), layer_4_semantic * 255)
#
#        save_Layer_4_fusion_root = os.path.join(save_path, 'Layer_4_fusion')
#        if not os.path.exists(save_Layer_4_fusion_root):
#            os.makedirs(save_Layer_4_fusion_root)
#        layer_4_fusion = F.upsample(fusion_list[3], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_4_fusion = layer_4_fusion.data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_4_fusion_root, name), layer_4_fusion * 255)
#
#        # save layer 5 edge, semantic, fusion feature map
#        save_Layer_5_edge_root = os.path.join(save_path, 'Layer_5_edge')
#        if not os.path.exists(save_Layer_5_edge_root):
#            os.makedirs(save_Layer_5_edge_root)
#        layer_5_edge = F.upsample(edge_list[4], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_5_edge = layer_5_edge.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_5_edge_root, name), layer_5_edge * 255)
#
#        save_Layer_5_semantic_root = os.path.join(save_path, 'Layer_5_semantic')
#        if not os.path.exists(save_Layer_5_semantic_root):
#            os.makedirs(save_Layer_5_semantic_root)
#        layer_5_semantic = F.upsample(sal_list[4], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_5_semantic = layer_5_semantic.sigmoid().data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_5_semantic_root, name), layer_5_semantic * 255)
#
#        save_Layer_5_fusion_root = os.path.join(save_path, 'Layer_5_fusion')
#        if not os.path.exists(save_Layer_5_fusion_root):
#            os.makedirs(save_Layer_5_fusion_root)
#        layer_5_fusion = F.upsample(fusion_list[4], size=gt.shape, mode='bilinear', align_corners=False)
#        layer_5_fusion = layer_5_fusion.data.cpu().numpy().squeeze()
#        cv2.imwrite(os.path.join(save_Layer_5_fusion_root, name), layer_5_fusion * 255)
        
        
    print('Test Done!')
   
#        res = res3
#        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
#        res = res.sigmoid().data.cpu().numpy().squeeze()
#        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#        print('save img to: ',save_path+name)

#        cv2.imwrite(save_path + name, res*255)
#    print('Test Done!')
