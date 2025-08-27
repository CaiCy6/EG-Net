import torch
import torch.nn as nn
import timm
import torch
from kornia.morphology import closing, opening
from kornia.filters import median_blur
import cv2
from torchvision.utils import save_image
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange


T_MAX = 96*96

from torch.utils.cpp_extension import load
wkv_cuda = load(name="wkv", sources=["/root/autodl-tmp/EG-Net/cuda/wkv_op.cpp", "/root/autodl-tmp/EG-Net/cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)


import torch
import torch.nn as nn
from einops import rearrange

class VRWKV_SpatialMix_Tri_Eff_2D(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        self.directions = 3 

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.fusion_conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.directions, self.n_embd)))
            self.spatial_first = nn.Parameter(torch.randn((self.directions, self.n_embd)))

    def shift_2d(self, input, shift_pixel=1, gamma=1/4, patch_resolution=None):

        B, N, C = input.shape
        H, W = patch_resolution
        # (B, N, C) -> (B, C, H, W)
        input = input.transpose(1, 2).reshape(B, C, H, W)
        output = torch.zeros_like(input)
        C_gamma = int(C * gamma)


        output[:, 0:C_gamma, :, shift_pixel:W] = input[:, 0:C_gamma, :, 0:W-shift_pixel]

        output[:, C_gamma:2*C_gamma, :, 0:W-shift_pixel] = input[:, C_gamma:2*C_gamma, :, shift_pixel:W]

        output[:, 2*C_gamma:3*C_gamma, shift_pixel:H, :] = input[:, 2*C_gamma:3*C_gamma, 0:H-shift_pixel, :]

        output[:, 3*C_gamma:4*C_gamma, 0:H-shift_pixel, :] = input[:, 3*C_gamma:4*C_gamma, shift_pixel:H, :]


        return output.flatten(2).transpose(1, 2)

    def jit_func(self, x, resolution):
        h, w = resolution
        x = self.shift_2d(input=x, patch_resolution=resolution)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution)
        h, w = resolution

        # 第一个方向：原顺序
        v1 = RUN_CUDA(B, T, C, self.spatial_decay[0] / T, self.spatial_first[0] / T, k, v)
        x = v1
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x

class VRWKV_ChannelMix_2D(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None

    def forward(self, x, resolution):
        h, w = resolution
        k = self.key(x)
        k = torch.square(torch.relu(k)) 
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(x))
        x = r * kv
        return x

class Block_2D(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4,
                 init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.att = VRWKV_SpatialMix_Tri_Eff_2D(n_embd, n_layer, layer_id, init_mode,
                                              key_norm=key_norm)
        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
    def forward(self, x):
        b, c, h, w = x.shape
        resolution = (h, w)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

class Block_2D_Channel(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4,
                 init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = VRWKV_ChannelMix_2D(n_embd, n_layer, layer_id, hidden_rate,
                                      init_mode, key_norm=key_norm)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
    def forward(self, x):
        b, c, h, w = x.shape
        resolution = (h, w)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

def extract_contour(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    输入: (B, 3, H, W), dtype=float32 [0~1]
    输出: (B, 1, H, W), dtype=float32 [0~1]，白色轮廓线
    """

    B, C, H, W = image_tensor.shape
    device = image_tensor.device

    # 结果保存列表
    contour_tensors = []
    sobel_tensors = []

    for b in range(B):

        img_np = (image_tensor[b].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        lower_skin = np.array([0, 20, 20])
        upper_skin = np.array([25, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # 可调整大小控制扩张程度
        mask_hsv_dilated = cv2.dilate(mask_hsv, kernel_dilate, iterations=1)
        mask_hsv = mask_hsv_dilated
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

        sobel_masked = cv2.bitwise_and(sobel_magnitude, sobel_magnitude, mask=mask_hsv)

        sobel_enhanced = np.uint8(np.power(sobel_masked / 255.0, 0.5) * 255)
        _, sobel_binary = cv2.threshold(sobel_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        sobel_binary = cv2.morphologyEx(sobel_binary, cv2.MORPH_CLOSE, kernel)


        contours, _ = cv2.findContours(sobel_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)

            contour_only = np.zeros_like(gray)
            cv2.drawContours(contour_only, [largest_contour], -1, color=255, thickness=3)

            contour_tensor = torch.from_numpy(contour_only.astype(np.float32) / 255.0).unsqueeze(0)  

        else:
            contour_tensor = torch.ones((1, H, W), dtype=torch.float32)

        sobel_binary_tensor = torch.from_numpy(sobel_binary.astype(np.float32) / 255.0).unsqueeze(0)  # shape: [1, H, W]

        contour_tensors.append(contour_tensor)
        sobel_tensors.append(sobel_binary_tensor) 
    # 合并所有 batch 的结果
    output_tensor = torch.stack(contour_tensors, dim=0).to(device)  # shape: [B, 1, H, W]
    sobel_tensor = torch.stack(sobel_tensors, dim=0).to(device)     # shape: [B, 1, H, W]

    return output_tensor,sobel_tensor


# class ImprovedChannelAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction_ratio=16):
#         super(ImprovedChannelAttention, self).__init__()
#         self.out_channels = out_channels
#         self.reduction_ratio = reduction_ratio
#         self.reduced_channels = max(out_channels // reduction_ratio, 1)

#         # 双路径压缩：均值 + 最大池化
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         # 共享权重的 MLP
#         self.mlp = nn.Sequential(
#             nn.Linear(out_channels, self.reduced_channels, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.reduced_channels, out_channels, bias=False)
#         )
#         # 可学习偏置项
#         self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
#         self.gamma = nn.Parameter(torch.ones(1, 1, 1, 1))
#         # 激活函数
#         self.sigmoid = nn.Sigmoid()
        
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.5),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
        
#     def forward(self, x):
        
#         x = self.decoder(x)
#         iden = x
#         b, c, h, w = x.size()
        
#         # 压缩（Squeeze）
#         avg_out = self.avg_pool(x).view(b, c)
#         max_out = self.max_pool(x).view(b, c)

#         # 激励（Excitation）共享MLP
#         avg_fc = self.mlp(avg_out).view(b, c, 1, 1)
#         max_fc = self.mlp(max_out).view(b, c, 1, 1)

#         # 合并 + sigmoid + bias
#         attention = self.sigmoid((avg_fc + max_fc) * self.gamma + self.bias)

#         # 应用注意力权重
#         return x * attention.expand_as(x) + iden

class ImprovedChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(ImprovedChannelAttention, self).__init__()
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.reduced_channels = max(out_channels // reduction_ratio, 1)

        # 双路径压缩：均值 + 最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享权重的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, self.reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduced_channels, out_channels, bias=False)
        )

        # 新增：1x1卷积分支（用于通道间建模）
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

        # 可学习偏置项
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        
        # 解码器部分保持不变
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.decoder(x)
        iden = x
        b, c, h, w = x.size()

        # 原有池化路径
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_fc = self.mlp(avg_out).view(b, c, 1, 1)
        max_fc = self.mlp(max_out).view(b, c, 1, 1)

        # 新增：1x1卷积路径（捕捉局部通道相关性）
        conv_path = self.bn1x1(self.conv1x1(x))
        conv_attention = F.adaptive_avg_pool2d(conv_path, 1)  # 再次池化用于 attention 融合

        # 合并所有路径
        attention = self.sigmoid((avg_fc + max_fc + conv_attention) * self.gamma + self.bias)

        return x * attention.expand_as(x) + iden

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=32,drop_rate=0.5):
        super(ResidualBlock2D, self).__init__()
        self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=drop_rate)

    def forward(self, x):
        identity = self.iden(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity  # 残差连接
        x = self.relu(x)
        return x

class ConvNeXtEGNet(nn.Module):
    def __init__(self, pretrained_encoder_path=None, pretrained=False, checkpoint_path=None,freeze_encoder=False):
        super(ConvNeXtEGNet, self).__init__()

        # 'convnext_small.in12k_ft_in1k_384'
        # 'convnextv2_tiny.fcmae'
        self.encoder = timm.create_model('convnext_small.in12k_ft_in1k_384', pretrained=False, features_only=True)

        if pretrained_encoder_path is not None:
            state_dict = torch.load(pretrained_encoder_path, map_location='cpu')
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                       if not k.startswith('head_') and not k.startswith('head.norm')}
            self.encoder.load_state_dict(filtered_state_dict)
        
        if pretrained:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path")
            state_dict = torch.load(checkpoint_path, map_location='cuda')
            self.load_state_dict(state_dict)
            self.to(device)
            self.eval()
            
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    
        # self.res = ResidualBlock2D()
        self.block0 = Block_2D(n_embd=96, n_layer=4, layer_id=0)
        self.block1 = Block_2D(n_embd=192, n_layer=4, layer_id=1)
        self.block2 = Block_2D(n_embd=384, n_layer=4, layer_id=2)
        self.block3 = Block_2D(n_embd=768, n_layer=4, layer_id=3)

        # self.decoder4 = self._decoder_block(768, 384)
        self.decoder4 = ImprovedChannelAttention(768, 384)
        self.decoder3 = ImprovedChannelAttention(384, 192)
        self.decoder2 = ImprovedChannelAttention(192, 96)
        self.decoder1 = ImprovedChannelAttention(96, 64)
        self.decoder0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.edge_sobel = nn.Conv2d(2, 1, kernel_size=1)
        
        self.global3 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.global2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.global1 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        
        self.EGBN2 = nn.BatchNorm2d(384)
        self.EGBN1 = nn.BatchNorm2d(192)
        self.EGBN0 = nn.BatchNorm2d(96)
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.att = nn.Sigmoid()
        # 定义 4 个可学习的标量权重，初始化为 0.1
        self.alpha0 = nn.Parameter(torch.tensor(0.1))
        self.alpha1 = nn.Parameter(torch.tensor(0.1))
        self.alpha2 = nn.Parameter(torch.tensor(0.1))
        self.alpha3 = nn.Parameter(torch.tensor(0.1))
        

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        hsv0,sobel = extract_contour(x)
        hsv0 = self.edge_sobel(torch.cat([hsv0,sobel],dim=1))
        
        hsv0 = self.pool(hsv0)
        hsv0 = self.pool(hsv0) # 56
        hsv1 = self.pool(hsv0) # 
        hsv2 = self.pool(hsv1)
        hsv3 = self.pool(hsv2)
        hsv0att = self.att(hsv0)
        hsv1att = self.att(hsv1)
        hsv2att = self.att(hsv2)
        hsv3att = self.att(hsv3)
        
        # res = self.res(x)
        
        feats = self.encoder(x)  # 输出4个阶段特征
        
        Edge_feats = [None] * 4
        Global_feats = [None] * 4
        
        # 下面是四层的Edge特征
        Edge_feats[0] = feats[0] + feats[0] * hsv0att * self.alpha0 # 96
        Edge_feats[1] = feats[1] + feats[1] * hsv1att * self.alpha1 # 48
        Edge_feats[2] = feats[2] + feats[2] * hsv2att * self.alpha2
        Edge_feats[3] = feats[3] + feats[3] * hsv3att * self.alpha3
        
        # 下面要写全局特征提取 SpatialMix部分
        Global_feats[0] = self.block0(feats[0])    # 96     96  6.5 B T C 96 96 96 
        Global_feats[1] = self.block1(feats[1])    # 192    48            192 48 48 
        Global_feats[2] = self.block2(feats[2])    # 384    24            384 24 24 
        Global_feats[3] = self.block3(feats[3])    # 768    12            768 12 12 

        # 下面写加权融合的模块
        feats[2] = self.EGBN2(feats[2] + Edge_feats[2] + self.global3(Global_feats[3]))
        feats[1] = self.EGBN1(feats[1] + Edge_feats[1] + self.global2(Global_feats[2]))
        feats[0] = self.EGBN0(feats[0] + Edge_feats[0] + self.global1(Global_feats[1]))        
        
        # feats[2] = self.EGBN2(feats[2] + Edge_feats[2] + self.global3(Global_feats[3]))
        # feats[1] = self.EGBN1(feats[1] + Edge_feats[1] + self.global2(Global_feats[2]))
        # feats[0] = self.EGBN0(feats[0] + Edge_feats[0] + self.global1(Global_feats[1]))

        # d4 = self.decoder4(feats[3])  # 从最深处上采样
        d4 = self.decoder4(feats[3])
        d3 = self.decoder3(d4 + feats[2])  # 简单拼接/加法
        d2 = self.decoder2(d3 + feats[1])
        d1 = self.decoder1(d2 + feats[0])
        
        d0 = self.decoder0(d1)        # 最后的一次上采样并且调整通道数为32
        out = self.final_conv(d0) # 加上残差块的输出并且调整通道数到输出类别 1  + res
        return out
    
if __name__ == '__main__':
    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count
    model = ConvNeXtEGNet().eval()  

    input_tensor = torch.randn(1, 3, 384, 384)  # batch_size=1, channel=3, HxW=256x256
    params = parameter_count(model)
    total_params = params["total"] / 1e6  
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total() / 1e9  
    print(f"Total Parameters: {total_params:.2f} M")
    print(f"Total FLOPs: {total_flops:.2f} G")
