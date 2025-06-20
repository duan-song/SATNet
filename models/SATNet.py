import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np
import torch.nn.functional as F
from models.SwinTransformers import SwinTransformer
from models.MobileNetV2 import mobilenet_v2
#from DeformableConvs import DeformConv2d
import datetime
import time


def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
        nn.BatchNorm2d(out_planes),
        nn.GELU(),
    )


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class NonLocal2d(nn.Module):
    def __init__(self, in_channels, reduction=2, use_scale=True, sub_sample=False, mode='embedded_gaussian'):
        super(NonLocal2d, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode

        if mode not in [
            'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError("Mode should be in 'gaussian', 'concatenation', "
                             f"'embedded_gaussian' or 'dot_product', but got "
                             f'{mode} instead.')

        self.g = nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1, )
        self.conv_out = nn.Conv2d(
            self.inter_channels,
            self.in_channels,
            kernel_size=1, )

        if self.mode != 'gaussian':
            self.theta = nn.Conv2d(
                self.in_channels,
                self.inter_channels,
                kernel_size=1, )
            self.phi = nn.Conv2d(
                self.in_channels,
                self.inter_channels,
                kernel_size=1, )

        if self.mode == 'concatenation':
            self.concat_project = nn.Sequential(
                nn.Conv2d(
                    self.inter_channels * 2,
                    1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.ReLU())
        self.sub_sample = sub_sample
        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer

        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        if self.mode != 'gaussian':
            for m in [self.g, self.theta, self.phi]:
                nn.init.normal_(m.weight.data, std=std)
        else:
            nn.init.normal_(self.g.weight.data, std=std)

        if zeros_init:
            nn.init.normal_(self.conv_out.weight.data, 0)
        else:
            nn.init.normal_(self.conv_out.weight.data, std=std)

    def gaussian(self, theta_x, phi_x):
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x, phi_x):
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x, phi_x):
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]

        return pairwise_weight

    def forward(self, x):
        # NonLocal2d x: [N, C, H, W]
        n = x.size(0)
        # NonLocal2d g_x: [N, HxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        # NonLocal2d theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # NonLocal2d y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # NonLocal2d y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *x.size()[2:])
        output = x + self.conv_out(y)

        return output


# Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), groups=out_channel, padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), groups=out_channel, padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, groups=out_channel, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), groups=out_channel, padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), groups=out_channel, padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, groups=out_channel, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), groups=out_channel, padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), groups=out_channel, padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, groups=out_channel, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.conv_trans = BasicConv2d(3 * in_channel, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, global_info, local_info):
        B, C, H, W = x.shape
        global_info = F.interpolate(global_info, size=(H, W), mode='bilinear')
        local_info = F.interpolate(local_info, size=(H, W), mode='bilinear')

        x = self.conv_trans(torch.cat([x, x * global_info, x * local_info], dim=1))

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class Patch_Attention(nn.Module):
    def __init__(self, in_channels, reduction=8, pool_window=10, add_input=False):
        super(Patch_Attention, self).__init__()
        self.pool_window = pool_window
        self.add_input = add_input
        self.SA = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        pool_h = h//self.pool_window
        pool_w = w//self.pool_window
        
        A = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
        A = self.SA(A)
        
        A = F.upsample(A, (h,w), mode='bilinear')        
        output = x*A
        if self.add_input:
            output += x
        
        return output

# Global Texture Module: GTM
class GTM(nn.Module):
    def __init__(self, inch, outch):
        super(GTM, self).__init__()
        self.pathconv = Patch_Attention(inch)
        self.dilateconv3 = nn.Conv2d(in_channels=inch, out_channels=inch, kernel_size=3, padding=3, stride=1,
                                     groups=inch, dilation=3)
        self.dilateconv6 = nn.Conv2d(in_channels=inch, out_channels=inch, kernel_size=3, padding=6, stride=1,
                                     groups=inch, dilation=6)
        self.dilateconv12 = nn.Conv2d(in_channels=inch, out_channels=inch, kernel_size=3, padding=12, stride=1,
                                     groups=inch, dilation=12)

        self.transconv = BasicConv2d(outch*4, outch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # identify = x
        x3 = self.dilateconv3(x)
        x6 = self.dilateconv6(x)
        x12 = self.dilateconv12(x)

        out = self.transconv(torch.cat([x, x3, x6, x12], dim=1))
        out = self.pathconv(out)
        return out


class SATNet(nn.Module):
    def __init__(self):
        super(SATNet, self).__init__()

        # self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        # self.depth_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.rgb_mobile = mobilenet_v2()
        self.depth_mobile = mobilenet_v2()
        # input 256*256*3
        # conv1 128*128*16
        # conv2 64*64*24
        # conv3 32*32*32
        # conv4 16*16*96
        # conv5 8*8*320
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.efficiency_scale = 32

        self.deconv_layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=self.efficiency_scale, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.deconv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=self.efficiency_scale, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.deconv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.efficiency_scale, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.deconv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=self.efficiency_scale, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.deconv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=self.efficiency_scale, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        self.rgbd_fusion_5 = CAAM(320, 320)
        self.rgbd_fusion_4 = CAAM(96, 96)
        self.rgbd_fusion_3 = CAAM(32, 32)
        self.rgbd_fusion_2 = CAAM(24, 24)
        self.rgbd_fusion_1 = CAAM(16, 16)

        self.conv_trans = nn.Conv2d(in_channels=2 * self.efficiency_scale, out_channels=self.efficiency_scale,
                                    kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_pre = nn.Conv2d(in_channels=self.efficiency_scale, out_channels=1, kernel_size=1, stride=1, padding=0,
                                  bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.selfatt = NonLocal2d(in_channels=self.efficiency_scale)

        self.refine_5 = GCM(self.efficiency_scale, self.efficiency_scale)
        self.refine_4 = GCM(self.efficiency_scale, self.efficiency_scale)
        self.refine_3 = GCM(self.efficiency_scale, self.efficiency_scale)
        self.refine_2 = GCM(self.efficiency_scale, self.efficiency_scale)
        self.refine_1 = GCM(self.efficiency_scale, self.efficiency_scale)

        self.texturerefine = GTM(self.efficiency_scale, self.efficiency_scale)

    def forward(self, x, d):
        #        rgb_list = self.rgb_swin(x)
        #        depth_list = self.depth_swin(d)

        start_time_encoder = time.time()
        rgb1, rgb2, rgb3, rgb4, rgb5 = self.rgb_mobile(x)
        depth1, depth2, depth3, depth4, depth5 = self.depth_mobile(d)
        end_time_encoder = time.time()
#        print("the backbone time is {}".format(end_time_encoder-start_time_encoder))

        start_time_dam = time.time()
        f5 = self.rgbd_fusion_5(rgb5, depth5)
        f4 = self.rgbd_fusion_4(rgb4, depth4)
        f3 = self.rgbd_fusion_3(rgb3, depth3)
        f2 = self.rgbd_fusion_2(rgb2, depth2)
        f1 = self.rgbd_fusion_1(rgb1, depth1)
        end_time_dam = time.time()
#        print("the DAM time is {}".format(end_time_dam-start_time_dam))

        start_time_cipn = time.time()
        d5_reduce = self.deconv_layer_5(f5)
        d4_reduce = self.deconv_layer_4(f4)
        d3_reduce = self.deconv_layer_3(f3)
        d2_reduce = self.deconv_layer_2(f2)
        d1_reduce = self.deconv_layer_1(f1)

        d5_att = torch.mean(d5_reduce, dim=1, keepdim=True)
        d4_att = torch.mean(d4_reduce, dim=1, keepdim=True)
        d3_att = torch.mean(d3_reduce, dim=1, keepdim=True)
        d2_att = torch.mean(d2_reduce, dim=1, keepdim=True)
        d1_att = torch.mean(d1_reduce, dim=1, keepdim=True)
        d_att = [d1_att, d2_att, d3_att, d4_att, d5_att]

        # the first feature pyramid from top to down
        pyramid_td_5 = d5_reduce
        pyramid_td_4 = self.conv_trans(torch.cat([self.up2(pyramid_td_5), d4_reduce], dim=1))
        pyramid_td_3 = self.conv_trans(torch.cat([self.up2(pyramid_td_4), d3_reduce], dim=1))
        pyramid_td_2 = self.conv_trans(torch.cat([self.up2(pyramid_td_3), d2_reduce], dim=1))
        pyramid_td_1 = self.conv_trans(torch.cat([self.up2(pyramid_td_2), d1_reduce], dim=1))

        ede_1 = torch.mean(pyramid_td_1, dim=1, keepdim=True)
        ede_2 = torch.mean(pyramid_td_2, dim=1, keepdim=True)
        ede_3 = torch.mean(pyramid_td_3, dim=1, keepdim=True)
        ede_4 = torch.mean(pyramid_td_4, dim=1, keepdim=True)
        ede_5 = torch.mean(pyramid_td_5, dim=1, keepdim=True)
        edge_list = [ede_1, ede_2, ede_3, ede_4, ede_5]

        # low-level texture information
        texture_info = self.texturerefine(pyramid_td_1)
        texture_pre = self.conv_pre(texture_info)

        # the second feature pyramid from down to top
        pyramid_dt_1 = pyramid_td_1
        pyramid_dt_2 = self.conv_trans(torch.cat([self.pool2(pyramid_dt_1), pyramid_td_2], dim=1))
        pyramid_dt_3 = self.conv_trans(torch.cat([self.pool2(pyramid_dt_2), pyramid_td_3], dim=1))
        pyramid_dt_4 = self.conv_trans(torch.cat([self.pool2(pyramid_dt_3), pyramid_td_4], dim=1))
        pyramid_dt_5 = self.conv_trans(torch.cat([self.pool2(pyramid_dt_4), pyramid_td_5], dim=1))

        sal_1 = torch.mean(pyramid_dt_1, dim=1, keepdim=True)
        sal_2 = torch.mean(pyramid_dt_2, dim=1, keepdim=True)
        sal_3 = torch.mean(pyramid_dt_3, dim=1, keepdim=True)
        sal_4 = torch.mean(pyramid_dt_4, dim=1, keepdim=True)
        sal_5 = torch.mean(pyramid_dt_5, dim=1, keepdim=True)
        sal_list = [sal_1, sal_2, sal_3, sal_4, sal_5]

        # high-level global semantic information
        semantic_info = self.selfatt(self.selfatt(self.selfatt(pyramid_dt_5)))
        semantic_pre = self.conv_pre(semantic_info)

        # feature refinement for enhancing feature representation
        decoder_5 = self.refine_5(pyramid_dt_5 + pyramid_td_5, semantic_info, texture_info)
        decoder_4 = self.refine_4(pyramid_dt_4 + pyramid_td_4, semantic_info, texture_info)
        decoder_3 = self.refine_3(pyramid_dt_3 + pyramid_td_3, semantic_info, texture_info)
        decoder_2 = self.refine_2(pyramid_dt_2 + pyramid_td_2, semantic_info, texture_info)
        decoder_1 = self.refine_1(pyramid_dt_1 + pyramid_td_1, semantic_info, texture_info)

        fusion_1 = torch.mean(decoder_1, dim=1, keepdim=True)
        fusion_2 = torch.mean(decoder_2, dim=1, keepdim=True)
        fusion_3 = torch.mean(decoder_3, dim=1, keepdim=True)
        fusion_4 = torch.mean(decoder_4, dim=1, keepdim=True)
        fusion_5 = torch.mean(decoder_5, dim=1, keepdim=True)
        fusion_list = [fusion_1, fusion_2, fusion_3, fusion_4, fusion_5]
        
        end_time_cipn = time.time()
#        print("the CIPN time is {}".format(end_time_cipn-start_time_cipn))

        out5_att = torch.mean(decoder_5, dim=1, keepdim=True)
        out4_att = torch.mean(decoder_4, dim=1, keepdim=True)
        out3_att = torch.mean(decoder_3, dim=1, keepdim=True)
        out2_att = torch.mean(decoder_2, dim=1, keepdim=True)
        out1_att = torch.mean(decoder_1, dim=1, keepdim=True)
        out_att = [out1_att, out2_att, out3_att, out4_att, out5_att]

        # decoder for saliency reasoning
        start_time_decoder = time.time()
        decoder_4 = self.conv_trans(torch.cat([self.up2(decoder_5), decoder_4], dim=1))
        decoder_3 = self.conv_trans(torch.cat([self.up2(decoder_4), decoder_3], dim=1))
        decoder_2 = self.conv_trans(torch.cat([self.up2(decoder_3), decoder_2], dim=1))
        decoder_1 = self.conv_trans(torch.cat([self.up2(decoder_2), decoder_1], dim=1))
        end_time_decoder = time.time()
#        print("the Decoder time is {}".format(end_time_decoder-start_time_decoder))

        pre = self.conv_pre(decoder_1)

        torch.mean(x, dim=1, keepdim=True)

        return self.up2(pre), self.up2(texture_pre), self.up4(self.up2(semantic_pre))

    # def load_pre(self, pre_model):
    #     self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
    #     print(f"RGB SwinTransformer loading pre_model ${pre_model}")
    #     self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
    #     print(f"Depth SwinTransformer loading pre_model ${pre_model}")


####################################################
# The implement by PyTorch for Coordinate Attention
####################################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=2):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# Cross-modal alternation attention module: CAAM
class CAAM(nn.Module):

    def __init__(self, inch, outch):
        super(CAAM, self).__init__()
        self.rgb_att = CoordAtt(inch, inch)
        self.depth_att = CoordAtt(inch, inch)
        self.self_SA_Enhance = SA_Enhance()
        self.conv_trnas = nn.Conv2d(inch * 2, outch, 1, 1, 0)

    def forward(self, rgb, depth):
        identify_rgb = rgb
        identify_depth = depth

        rgb_att = self.self_SA_Enhance(self.rgb_att(rgb))
        depth_att = self.self_SA_Enhance(self.depth_att(depth))

        identify_rgb = identify_rgb * depth_att
        identify_depth = identify_depth * rgb_att

        ful_mul = torch.mul(identify_rgb, identify_depth)
        x_in1 = torch.reshape(identify_rgb, [identify_rgb.shape[0], 1, identify_rgb.shape[1], identify_rgb.shape[2],
                                             identify_rgb.shape[3]])
        x_in2 = torch.reshape(identify_depth,
                              [identify_depth.shape[0], 1, identify_depth.shape[1], identify_depth.shape[2],
                               identify_depth.shape[3]])
        x_cat = torch.cat((x_in1, x_in2), dim=1)
        ful_max = x_cat.max(dim=1)[0]

        out = self.conv_trnas(torch.cat([ful_mul, ful_max], dim=1))
        return out


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x