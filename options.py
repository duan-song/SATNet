import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')  # 100
parser.add_argument('--load_pre', type=str, default='/media/gpu/dss/dataset/SOD/SwinTransformer/swin_base_patch4_window12_384_22k.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='../train/RGB/', help='the training rgb images root')  # train_dut
parser.add_argument('--depth_root', type=str, default='../train/Depth_pseudo/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='../train/GT/', help='the training gt images root')
parser.add_argument('--edge_root', type=str, default='../train/Edge/', help='the training rgb texture root')  # train_dut
parser.add_argument('--test_rgb_root', type=str, default='../validation/RGBD135/RGB/', help='the test gt images root')
parser.add_argument('--test_depth_root', type=str, default='../validation/RGBD135/Depth_pseudo/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='../validation/RGBD135/GT/', help='the test gt images root')
# parser.add_argument('--test_texture_root', type=str, default='/media/gpu/dss/dataset/SOD/SwinTransformer/validation/RGBD135/Texture/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default='./cpts/', help='the path to save models and logs')
opt = parser.parse_args()