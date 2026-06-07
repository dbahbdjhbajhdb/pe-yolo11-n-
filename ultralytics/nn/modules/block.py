# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn
from einops import rearrange
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
    "RFAConv",
    "DynamicAttention",
    "HDRAB",
    "RHDWT",
)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size  # 卷积核大小

        # 获取权重的网络结构
        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),  # 平均池化层
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                      groups=in_channel, bias=False)  # 卷积层，用于生成权重
        )

        # 生成特征的网络结构
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size,
                      padding=kernel_size // 2, stride=stride, groups=in_channel, bias=False),  # 卷积层，用于生成特征
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),  # 批归一化层
            nn.ReLU()  # 激活函数
        )

        # 最终的卷积层
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]  # 获取输入张量的批量大小和通道数
        weight = self.get_weight(x)  # 生成权重
        h, w = weight.shape[2:]  # 获取权重张量的高度和宽度
        # 对权重进行reshape并应用softmax
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)
        # 生成特征并进行reshape
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)
        # 对特征和权重进行乘法操作
        weighted_data = feature * weighted
        # 使用einops库对张量进行重排
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)
        # 应用最终的卷积层
        return self.conv(conv_data)
import torch
import torch.nn as nn
import math
from einops import rearrange

try:
    from ultralytics.nn.modules.conv import Conv
except ImportError:
    class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

class StandardRFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # -------------------------------------------------------------
        # 💡 关键修改：动态计算分组数 (g)
        # 我们希望每组大约有 8~16 个通道进行交流，而不是全通道交流。
        # 这样可以将参数量降低 8~16 倍，同时保证足够的信息融合。
        # -------------------------------------------------------------

        # 目标：每组负责 8 个通道 (参数量变为原来的 1/8)
        channels_per_group = 8
        # 确保 groups 能被 in_channel 整除
        g = math.gcd(in_channel, in_channel // channels_per_group)
        if g == 0: g = 1  # 防止特殊情况

        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),

            # ---> 轻量级通道融合 (Grouped Pointwise Conv) <---
            # 参数量 = (C * C) / g
            # 相比全连接 1x1，这里节省了 g 倍的参数
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, groups=g, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.SiLU(),

            # 生成权重 (保持 Depthwise 以极致省参数)
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                      groups=in_channel, bias=False)
        )

        # 特征生成部分保持不变 (Depthwise)，这是大头，必须省
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size,
                      padding=kernel_size // 2, stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU()
        )

        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]

        weight = self.get_weight(x)
        h, w = weight.shape[2:]

        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)

        weighted_data = feature * weighted

        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)',
                              n1=self.kernel_size, n2=self.kernel_size)

        return self.conv(conv_data)


import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class RHDWT(nn.Module):
    """
    残差离散小波变换下采样模块 (RHDWT)

    兼容性修复版：
    允许像 nn.Conv2d 那样传入 kernel_size 参数（会被自动忽略），
    从而避免因直接替换 Conv2d 导致的 "int object has no attribute lower" 错误。
    """

    def __init__(self, in_channels, out_channels, kernel_size=None, stride=2, wavelet='db2'):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: (可选) 仅用于兼容 nn.Conv2d 的调用格式，实际不使用，可传 3 或 None
            stride: 步长，默认为 2
            wavelet: 小波类型，默认为 'db2'
        """
        super().__init__()

        # --- 兼容性逻辑开始 ---
        # 如果用户把 'db2' 传到了 kernel_size 的位置 (也就是第3个位置)，我们要纠正过来
        if isinstance(kernel_size, str):
            wavelet = kernel_size
        # --- 兼容性逻辑结束 ---

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.wavelet = wavelet

        # 1. 获取小波滤波器并构建 2D 卷积核
        self.register_buffer('wavelet_filter', self._get_wavelet_filter(wavelet, in_channels))

        # 2. 残差连接后的特征融合层
        # stride设为1，因为小波变换步骤已经完成了下采样
        self.res_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False
        )

    def _get_wavelet_filter(self, wavelet_name, in_channels):
        """生成兼容 PyTorch Conv2d 的小波核"""
        try:
            w = pywt.Wavelet(wavelet_name)
        except ValueError:
            raise ValueError(f"无法识别小波名称: {wavelet_name}。请确保你已安装 PyWavelets 且名称正确。")
        except AttributeError:
            # 双重保险，防止参数错乱
            raise ValueError(f"参数传递错误：wavelet 期望是字符串，但收到了 {type(wavelet_name)}。")

        # PyTorch 的 Conv2d 是互相关，为了实现卷积性质，需要翻转滤波器
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)

        # 生成 2D 滤波器 (外积)
        ll = torch.outer(dec_lo, dec_lo)  # Low-Low
        lh = torch.outer(dec_lo, dec_hi)  # Low-High
        hl = torch.outer(dec_hi, dec_lo)  # High-Low
        hh = torch.outer(dec_hi, dec_hi)  # High-High

        # 堆叠所有滤波器 [4, K, K]
        filters = torch.stack([ll, lh, hl, hh], dim=0)

        # 扩展为 Depthwise Conv 权重: [In_C * 4, 1, K, K]
        filters = filters.repeat(in_channels, 1, 1, 1)

        return filters

    def forward(self, x):
        B, C, H, W = x.shape

        # padding 计算：针对 db2 (len=4), stride=2，padding=1 保持 H/2 输出
        # 如果你换了其他长度的小波，需要调整这里的 padding
        padding = 1 if self.wavelet == 'db2' else 0

        # 1. 小波变换 (使用分组卷积实现 Depthwise)
        dwt_out = F.conv2d(x, self.wavelet_filter, stride=self.stride, padding=padding, groups=C)

        # 重塑维度以分离 4 个频带: [B, C, 4, H/2, W/2]
        dwt_out = dwt_out.view(B, C, 4, H // self.stride, W // self.stride)

        # 分离频带
        ll = dwt_out[:, :, 0, :, :]  # 近似分量
        lh = dwt_out[:, :, 1, :, :]  # 水平细节
        hl = dwt_out[:, :, 2, :, :]  # 垂直细节
        # hh = dwt_out[:, :, 3, :, :] # 对角细节

        # 2. 残差高频反哺
        fused = ll + 0.5 * (lh + hl)

        # 3. 特征重组
        out = self.res_conv(fused)

        return out


# --- 验证部分 ---
if __name__ == "__main__":
    # 创建模拟输入
    x = torch.randn(2, 64, 128, 128)

    print("正在测试兼容性...")

    # 场景 1: 你的旧代码风格 (带 kernel_size=3) -> 现在应该能正常运行了
    # 这里的 '3' 会被自动赋给 kernel_size 参数并被忽略
    model_legacy = RHDWT(64, 128, 3, stride=2)
    y1 = model_legacy(x)
    print(f"场景1 (带参3) 输出尺寸: {y1.shape}")  #

class DynamicAttention(nn.Module):
    """
    DAU-YOLO 核心模块 (防呆修正版)
    --------------------------------
    修复: 强制 kernel_size=3，防止 YOLO 解析器将通道数(32/64)误传为核大小，
          从而引发 "shape invalid for input of size 1089" 错误。
    """

    # 1. 修改 init 签名，增加 *args 吸收多余参数
    def __init__(self, c1, c2, *args, kernel_size=3, reduction=4):
        super().__init__()

        # === 核心修复: 强制锁定参数 ===
        # 无论 YAML 传进来什么奇怪的数字 (如 32, 64)，我们只认 3
        # 这能 100% 解决 1089 vs 1024 的报错
        self.kernel_size = 3

        self.c1 = c1
        self.reduction = int(reduction)
        self.N = self.kernel_size * self.kernel_size  # N=9

        # === 定义层 ===
        # 偏移生成器 (输出 3*9=27 通道)
        self.conv_offset = nn.Conv2d(c1, 3 * self.N, kernel_size=3, padding=1)
        # 采样权重
        self.weight = nn.Parameter(torch.zeros(c1, self.N))

        # DyReLU 参数生成器
        hidden = max(c1 // self.reduction, 4)
        self.fc1 = nn.Linear(c1, hidden)
        self.fc2 = nn.Linear(hidden, c1 * 4)

        # 注册基准网格 (FP32)
        base_range = self.kernel_size // 2  # 3//2 = 1
        # 生成范围: -1, 0, 1
        base_offsets = [[dy, dx] for dy in range(-base_range, base_range + 1) for dx in
                        range(-base_range, base_range + 1)]
        self.register_buffer("base_offsets", torch.tensor(base_offsets, dtype=torch.float32), persistent=False)

        # === 初始化权重 ===
        self._init_weights()

    def _init_weights(self):
        # Offset 初始化: 微小随机数打破对称性
        nn.init.normal_(self.conv_offset.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv_offset.bias, 0.0)

        # Mask 初始化: 偏置设为 3.0 (Sigmoid后接近1)，初始开启
        center_mask_idx = 2 * self.N + (self.N // 2)
        with torch.no_grad():
            self.conv_offset.bias.data[2 * self.N:] = 3.0

            # DyReLU 初始化
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)

        # Leaky 模式: alpha1=1, alpha2=0.25
        with torch.no_grad():
            bias_init = torch.zeros(self.c1 * 4)
            bias_init[0:self.c1] = 1.0
            bias_init[2 * self.c1:3 * self.c1] = 0.25
            self.fc2.bias = nn.Parameter(bias_init)

            # 采样中心点权重设为 1
            if self.N > 0:
                self.weight.data[:, self.N // 2] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        dtype, device = x.dtype, x.device

        # FP32 精度保护
        x_fp32 = x.float()

        # 1. 生成偏移 (B, 3N, H, W)
        offset_out = self.conv_offset(x).float()

        offset_x = offset_out[:, 0:self.N]
        offset_y = offset_out[:, self.N:2 * self.N]
        mask = offset_out[:, 2 * self.N:3 * self.N].sigmoid()

        # 2. 构建网格
        base_y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        base_x = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
        bo = self.base_offsets.to(device=device, dtype=torch.float32)

        # 3. 绝对坐标
        ty = base_y + bo[:, 0].view(1, self.N, 1, 1) + offset_y
        tx = base_x + bo[:, 1].view(1, self.N, 1, 1) + offset_x

        # 4. 归一化 (Align Corners = False)
        ny = (2.0 * ty + 1.0) / max(H, 1) - 1.0
        nx = (2.0 * tx + 1.0) / max(W, 1) - 1.0
        grid = torch.stack((nx, ny), dim=-1)

        # 5. 采样
        out_fp32 = torch.zeros_like(x_fp32)
        for n in range(self.N):
            sampled = F.grid_sample(x_fp32, grid.select(1, n), mode="bilinear", padding_mode="zeros",
                                    align_corners=False)
            w_n = self.weight[:, n].view(1, C, 1, 1).float()
            m_n = mask[:, n].unsqueeze(1)
            out_fp32 += sampled * w_n * m_n

        out = out_fp32.to(dtype)

        # 6. TAA (DyReLU)
        ctx = out.mean(dim=(2, 3))
        theta = self.fc2(F.relu(self.fc1(ctx)))

        a1 = theta[:, 0:C].view(B, C, 1, 1)
        b1 = theta[:, C:2 * C].view(B, C, 1, 1)
        a2 = theta[:, 2 * C:3 * C].view(B, C, 1, 1)
        b2 = theta[:, 3 * C:4 * C].view(B, C, 1, 1)

        return torch.maximum(a1 * out + b1, a2 * out + b2)


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.SiLU()
        # self.nonlinearity = nn.ReLU()
        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_1x1'):
            kernel, bias = self.get_equivalent_kernel_bias()
            self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                         out_channels=self.rbr_dense.conv.out_channels,
                                         kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                         padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                         groups=self.rbr_dense.conv.groups, bias=True)
            self.rbr_reparam.weight.data = kernel
            self.rbr_reparam.bias.data = bias
            for para in self.parameters():
                para.detach_()
            self.rbr_dense = self.rbr_reparam
            # self.__delattr__('rbr_dense')
            self.__delattr__('rbr_1x1')
            if hasattr(self, 'rbr_identity'):
                self.__delattr__('rbr_identity')
            if hasattr(self, 'id_tensor'):
                self.__delattr__('id_tensor')
            self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def forward(self, inputs):
        if self.deploy:
            return self.nonlinearity(self.rbr_dense(inputs))
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))



import torch
from torch import nn

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor  # 分组数，默认为32
        assert channels // self.groups > 0  # 确保通道数能够被分组数整除
        self.softmax = nn.Softmax(-1)  # 定义 Softmax 层，用于最后一维度的归一化
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化，将特征图缩小为1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 自适应平均池化，保留高度维度，将宽度压缩为1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 自适应平均池化，保留宽度维度，将高度压缩为1
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 分组归一化
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)  # 1x1卷积
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)  # 3x3卷积

    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入张量的尺寸：批次、通道、高度、宽度
        group_x = x.reshape(b * self.groups, -1, h, w)  # 将张量按组重构：批次*组数, 通道/组数, 高度, 宽度
        x_h = self.pool_h(group_x)  # 对高度方向进行池化，结果形状为 (b*groups, c//groups, h, 1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 对宽度方向进行池化，并转置结果形状为 (b*groups, c//groups, 1, w)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 将池化后的特征在高度方向拼接后进行1x1卷积
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 将卷积后的特征分为高度特征和宽度特征
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 结合高度和宽度特征，应用分组归一化
        x2 = self.conv3x3(group_x)  # 对重构后的张量应用3x3卷积
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x1 进行自适应平均池化并应用Softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 重构 x2 的形状为 (b*groups, c//groups, h*w)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 进行自适应平均池化并应用Softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 重构 x1 的形状为 (b*groups, c//groups, h*w)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  # 计算权重，并重构为 (b*groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)  # 将权重应用于原始张量，并重构为原始输入形状




class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class CPCA(nn.Module):
    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out


if __name__ == '__main__':
    x = torch.randn(4, 64, 128, 128).cuda()
    model = CPCA(64, 64).cuda()
    out = model(x)
    print(out.shape)

import torch
import torch.nn as nn
import torchvision.ops


class RDAttention(nn.Module):
    """
    [RDAttention 终极完全体]
    集成特性：
    1. Multi-Head Spatial Diffusion: 多头注意力机制，独立学习不同部位的形变。
    2. Context-Aware Perception: 引入 5x5 上下文感知层，增强偏移量的准确性。
    3. Full-Capacity Task-Aware: 任务感知模块不进行降维 (reduction=1)，最大化多任务适应能力。
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, num_groups=4):
        super().__init__()

        # ============================================================
        # Part 1: 多头上下文感知空间扩散 (Multi-Head Context-Aware Spatial)
        # ============================================================
        if p is None:
            p = k // 2

        self.k = k
        self.s = s
        self.p = p
        self.num_groups = num_groups
        self.out_channels = c2

        # 确保通道数能被组数整除 (通常 c1=c2)
        # 如果 c1 != c2，这里假设 c1 是输入，c2 是输出
        assert c1 % num_groups == 0, f"输入通道数 {c1} 必须能被组数 {num_groups} 整除"
        assert c2 % num_groups == 0, f"输出通道数 {c2} 必须能被组数 {num_groups} 整除"

        self.group_in_channels = c1 // num_groups
        self.group_out_channels = c2 // num_groups

        # 1.1 上下文感知层 (Context Perception)
        # 在计算 Offset 前，先用 5x5 DWConv 提取大范围上下文信息
        self.offset_context = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=5, padding=2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )

        # 1.2 多头偏移与掩码生成器 (Multi-Head Generators)
        # 使用 ModuleList 存储每一组的生成器
        self.offset_convs = nn.ModuleList([
            nn.Conv2d(self.group_in_channels, 2 * k * k, kernel_size=k, stride=s, padding=p)
            for _ in range(num_groups)
        ])

        self.mask_convs = nn.ModuleList([
            nn.Conv2d(self.group_in_channels, k * k, kernel_size=k, stride=s, padding=p)
            for _ in range(num_groups)
        ])

        # 1.3 多头 DCN 权重参数
        # 形状: [num_groups, group_out, group_in, k, k]
        self.weight = nn.Parameter(
            torch.empty(num_groups, self.group_out_channels, self.group_in_channels // g, k, k)
        )
        self.bias = nn.Parameter(torch.empty(c2))

        # 1.4 融合层 (Projection)
        # 将多头的结果融合
        self.proj = nn.Conv2d(c2, c2, kernel_size=1)

        # ============================================================
        # Part 2: 归一化 (Normalization)
        # ============================================================
        self.bn = nn.BatchNorm2d(c2)

        # ============================================================
        # Part 3: 满血版任务感知 (Full-Capacity Task-Aware)
        # ============================================================
        # reduction=1 表示不压缩通道，参数量最大，特征保留最完整
        reduction = 1
        self.task_k = 2  # max(a1x+b1, a2x+b2)

        # Hyper-function 生成器
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c2, c2 // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        # 输出参数: 2 * k * channels (对应每个通道的 α1, β1, α2, β2)
        self.fc2 = nn.Conv2d(c2 // reduction, 2 * self.task_k * c2, kernel_size=1)

        # 初始化所有参数
        self.reset_parameters()

    def reset_parameters(self):
        # DCN 权重初始化
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

        # Offset/Mask 初始化为 0
        for m in self.offset_convs:
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)
        for m in self.mask_convs:
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)

        # Projection 初始化
        nn.init.kaiming_uniform_(self.proj.weight, a=1)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

        # Task-Aware 初始化 (接近普通 ReLU)
        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape

        # -----------------------------------------------
        # Step 1: 上下文感知 + 多头空间扩散
        # -----------------------------------------------
        # A. 上下文感知
        ctx = self.offset_context(x)

        # B. 分组 (Split)
        # x_groups: List of [B, C/G, H, W]
        x_groups = torch.chunk(x, self.num_groups, dim=1)
        ctx_groups = torch.chunk(ctx, self.num_groups, dim=1)

        out_groups = []

        # C. 对每一组独立进行 DCN
        for i in range(self.num_groups):
            # 使用感知后的特征计算 Offset 和 Mask
            offset = self.offset_convs[i](ctx_groups[i])
            mask = torch.sigmoid(self.mask_convs[i](ctx_groups[i]))

            # DCN 运算
            out_i = torchvision.ops.deform_conv2d(
                input=x_groups[i],
                offset=offset,
                weight=self.weight[i],
                bias=None,  # Bias 最后统一加
                stride=self.s,
                padding=self.p,
                mask=mask
            )
            out_groups.append(out_i)

        # D. 拼接 (Concat)
        out = torch.cat(out_groups, dim=1)

        # E. 加 Bias
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        # F. 融合投影
        out = self.proj(out)

        # -----------------------------------------------
        # Step 2: BN
        # -----------------------------------------------
        out = self.bn(out)

        # -----------------------------------------------
        # Step 3: 任务感知动态激活
        # -----------------------------------------------
        # 计算动态系数 theta
        theta = self.avg_pool(out)  # (B, C, 1, 1)
        theta = self.relu(self.fc1(theta))
        theta = self.fc2(theta)  # (B, 4C, 1, 1) [因为reduction=1, k=2, 所以是 2*2*C = 4C]

        # Reshape: (B, C, 2*k) -> (B, C, k, 2)
        theta = theta.view(b, c, 2 * self.task_k)
        theta = theta.view(b, c, self.task_k, 2)

        # 提取 alpha, beta
        alphas = theta[..., 0].view(b, c, self.task_k, 1, 1)
        betas = theta[..., 1].view(b, c, self.task_k, 1, 1)

        # 执行 max(a1*x+b1, a2*x+b2)
        x_expand = out.unsqueeze(2)  # (B, C, 1, H, W)
        res = alphas * x_expand + betas
        res, _ = torch.max(res, dim=2)

        return res


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops


class SpatialDiffusionBlock(nn.Module):
    """
    [Spatial-Diffusion Block]
    完全修复版: 保证 offset_scale 被正确定义，防止 AttributeError。
    """

    def __init__(self, in_channels, k=3, s=1, p=1):
        super().__init__()
        # 1. 上下文引导
        self.context_guide = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        # 2. 偏移量生成
        self.offset_conv = nn.Conv2d(in_channels, 2 * k * k, kernel_size=k, stride=s, padding=p)
        # 3. 掩码生成
        self.mask_conv = nn.Conv2d(in_channels, k * k, kernel_size=k, stride=s, padding=p)

        # 4. DCN 权重
        # 注意：这里将 groups 设置为 1，确保和 standard DCN 兼容
        self.weight = nn.Parameter(torch.empty(in_channels, in_channels, k, k))
        self.bias = nn.Parameter(torch.empty(in_channels))

        # ============================================================
        # 🔑 关键修复：必须在调用 reset_parameters 之前定义它！
        # ============================================================
        self.offset_scale = nn.Parameter(torch.tensor(1.0))

        self.stride = s
        self.padding = p

        # 最后再初始化参数，防止访问不到上面的变量
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        nn.init.constant_(self.mask_conv.weight, 0)
        nn.init.constant_(self.mask_conv.bias, 0)

        # 如果你的代码里还有这行，现在它不会报错了，因为我们在 __init__ 里定义了它
        if hasattr(self, 'offset_scale'):
            nn.init.constant_(self.offset_scale, 0.5)

    def forward(self, x):
        guidance = self.context_guide(x)

        # 应用 offset_scale 控制偏移幅度
        offset = self.offset_conv(guidance) * self.offset_scale

        mask = torch.sigmoid(self.mask_conv(guidance))

        return torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask
        )


class TaskAwareBlock(nn.Module):
    """
    [Context-Guided Task-aware Block]
    修正版特点：
    1. 使用 5x5 DW-Conv 获取局部上下文 (Local Context)。
    2. 移除 Gumbel，使用稳定 Softmax。
    3. 全卷积生成像素级参数，无全局池化。
    """

    def __init__(self, channels, k=2):
        super().__init__()
        self.k = k

        # -----------------------------------------------------------
        # [核心修正] 参数生成器
        # -----------------------------------------------------------
        self.theta_generator = nn.Sequential(
            # 第一步：大核深度卷积 (5x5 DWConv)
            # 作用：感受野扩大，能看清像素周围的环境，判断是不是物体
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            # 第二步：1x1 卷积生成参数
            # 输入: C -> 输出: 2 * k * C
            nn.Conv2d(channels, 2 * k * channels, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 生成具备上下文信息的动态参数
        # theta shape: [B, 2*k*C, H, W]
        theta = self.theta_generator(x)

        # 2. 拆分 Alpha 和 Beta
        theta = theta.view(b, c, self.k, 2, h, w)
        alphas = theta[:, :, :, 0, :, :]  # [B, C, k, H, W]
        betas = theta[:, :, :, 1, :, :]  # [B, C, k, H, W]

        # 3. 计算多分支特征
        x_expand = x.unsqueeze(2)
        output = alphas * x_expand + betas

        # 4. Softmax 加权 (回归稳定)
        # 这里的 Softmax 是针对 k 个分支的
        # 先减去最大值保证数值稳定
        output_max = output.max(dim=2, keepdim=True)[0]
        logits = output - output_max
        probs = F.softmax(logits, dim=2)

        # 5. 加权融合
        final_output = (probs * output).sum(dim=2)

        return final_output

class DAUBlock(nn.Module):
    """
    [DAU 完整模块]
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1):
        super().__init__()
        if p is None: p = k // 2

        self.c1 = c1
        self.c2 = c2

        # 1. Spatial Attention
        self.spatial = SpatialDiffusionBlock(c1, k=k, s=s, p=p)

        # 2. Norm
        self.bn = nn.BatchNorm2d(c1)

        # 3. Task Attention (Context-Guided)
        # 不需要传 total_steps 了，因为去掉了退火
        self.task = TaskAwareBlock(c1, k=2)

        # 4. Projection
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        identity = x

        x = self.spatial(x)
        x = self.bn(x)
        x = self.task(x)

        if self.c1 == self.c2:
            x = x + identity

        return self.proj(x)

# ----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops


# ================================================================
#  模块 1: SpatialDiffusionBlock (保持不变)
#  负责空间对齐，解决小目标“对不准”的问题
# ================================================================

class SpatialDiffusionBlock(nn.Module):
    def __init__(self, in_channels, k=3, s=1, p=1, offset_scale=2.0, mask_bias_init=2.0):
        super().__init__()
        self.k = k
        self.stride = s
        self.padding = p
        self.offset_scale = float(offset_scale)

        self.context_guide = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        self.offset_conv = nn.Conv2d(in_channels, 2 * k * k, kernel_size=k, stride=s, padding=p)
        self.mask_conv = nn.Conv2d(in_channels, k * k, kernel_size=k, stride=s, padding=p)
        self.weight = nn.Parameter(torch.empty(in_channels, in_channels, k, k))
        self.bias = nn.Parameter(torch.empty(in_channels))

        self.reset_parameters(mask_bias_init)

    def reset_parameters(self, mask_bias_init):
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.constant_(self.mask_conv.weight, 0)
        nn.init.constant_(self.mask_conv.bias, float(mask_bias_init))

    def forward(self, x):
        guidance = self.context_guide(x)
        offset = torch.tanh(self.offset_conv(guidance)) * self.offset_scale
        mask = torch.sigmoid(self.mask_conv(guidance))

        return torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask
        )


# ================================================================
#  模块 2: ChannelDeNoisingBlock (不压缩通道版)
#  修改点: 移除了 reduction，中间层通道数 = 输入通道数
# ================================================================

class ChannelDeNoisingBlock(nn.Module):
    """
    [Full-Rank Channel De-noising Block]
    1. Spike Suppression: 局部平滑去噪。
    2. Dual-Pool: Max+Avg 双路感知。
    3. No-Reduction: 全通道交互，不设瓶颈，保留所有特征细节。
    4. Soft Thresholding: 物理关闭噪音通道。
    """

    def __init__(self, channels, threshold=0.1):
        # 注意：这里删除了 reduction 参数
        super().__init__()
        self.threshold = float(threshold)

        # [关键修改] 不压缩通道
        # 保持 mid_channels 与 channels 一致
        # 对于 Nano 模型 (C=64/128)，这只会增加少量参数，但能大幅提升特征表达能力
        mid_channels = channels

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid_channels),  # C -> C
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels)  # C -> C
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 尖峰抑制 (Spike Suppression)
        x_smoothed = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # 2. 提取双路指纹
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x_smoothed, 1))

        # 3. 融合决策
        raw_scores = avg_out + max_out
        scale = torch.sigmoid(raw_scores).view(b, c, 1, 1)

        # 4. 软阈值去噪
        zero_mask = F.relu(scale - self.threshold)
        final_scale = zero_mask / (1 - self.threshold + 1e-6)

        return x * final_scale


# ================================================================
#  模块 3: SCBlock (完整封装)
# ================================================================

class SCBlock(nn.Module):
    """
    [Spatial-Channel Block | Full Rank]
    """

    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k // 2

        self.c1 = c1
        self.c2 = c2

        # 1. Spatial Focus
        self.spatial = SpatialDiffusionBlock(c1, k=k, s=s, p=p, offset_scale=2.0)

        self.bn = nn.BatchNorm2d(c1)

        # 2. Channel Cleaning (无压缩版)
        # 移除了 reduction 参数传递
        self.channel_cleaner = ChannelDeNoisingBlock(c1, threshold=0.15)

        # 3. Projection & Residual
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        self.res_proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x):
        identity = x

        # 空间对齐
        x = self.spatial(x)
        x = self.bn(x)

        # 通道去噪
        x = self.channel_cleaner(x)

        # 残差连接
        return self.proj(x) + self.res_proj(identity)


if __name__ == "__main__":
    # 测试维度匹配和参数量
    x = torch.randn(2, 64, 64, 64)  # 模拟 Nano 的 P3 层
    model = SCBlock(64, 64)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    # 打印 MLP 结构确认无压缩
    print("\nCheck MLP structure (Should be 64 -> 64):")
    print(model.channel_cleaner.mlp)


# ================================================================
#  简单测试 (Sanity Check)
# ================================================================
import torch
import torch.nn as nn


# 假设你已经定义了这两个基础模块
# from .your_module import SpatialDiffusionBlock, ChannelDeNoisingBlock

class ChannelRanker(nn.Module):
    """
    [通道评分器]
    基于 GAP + MLP 给通道打分，用于判断通道是否为"噪音"。
    """

    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.scorer = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1, bias=False),
            nn.Sigmoid()  # 输出 0~1 的分值
        )

    def forward(self, x):
        # x: [B, C, H, W] -> scores: [B, C, 1, 1]
        return self.scorer(self.avg_pool(x))

class SDC_Selective_Fusion(nn.Module):
    """
    [SDC 动态筛选融合模块]
    Inputs: [Deep_Feature(LowRes), Current_Feature(HighRes)]
    Logic:
      1. Align & Upsample Deep Feature.
      2. Spatial Fusion -> SpatialDiffusionBlock.
      3. Channel Fusion -> ChannelDeNoisingBlock.
      4. Ranker: 给去噪后的通道打分。
      5. Selector: 物理保留分数最高的 c2 个通道 (通常 c2 = c_in / 2)。
    """

    def __init__(self, c1, c2):
        super().__init__()
        # c1 是 list: [deep_ch, current_ch]
        # c2 是 目标输出通道数 (即筛选后保留的通道数)
        c_deep, c_current = c1[0], c1[1]
        self.out_channels = c2

        # 1. 对齐深层特征 (Deep Alignment)
        self.align_conv = nn.Conv2d(c_deep, c_current, 1, 1, 0, bias=False)
        self.bn_align = nn.BatchNorm2d(c_current)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # 2. 空间融合路径 (Spatial Path)
        # Concat(UP(Deep), Current) -> 2*C -> C
        self.spatial_fusion_conv = nn.Conv2d(c_current * 2, c_current, 1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(c_current)
        self.act = nn.SiLU()
        # 你的空间模块
        self.spatial_diffusion = SpatialDiffusionBlock(c_current)

        # 3. 通道融合路径 (Channel Path)
        # Concat(Spatial_Out, Current) -> 2*C -> C
        self.channel_fusion_conv = nn.Conv2d(c_current * 2, c_current, 1, bias=False)
        self.channel_bn = nn.BatchNorm2d(c_current)
        # 你的去噪模块
        self.channel_cleaner = ChannelDeNoisingBlock(c_current)

        # 4. 评分器 (Ranker)
        self.ranker = ChannelRanker(c_current)

        # 5. 最终归一化 (针对筛选后的特征)
        self.bn_final = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        # x: [x_deep, x_current]
        x_ds, x_ca = x[0], x[1]

        # --- Step 1: Deep Feature Upsampling ---
        ds_aligned = self.bn_align(self.align_conv(x_ds))
        ds_up = self.upsample(ds_aligned)

        # 尺寸对齐
        if ds_up.size()[-2:] != x_ca.size()[-2:]:
            ds_up = torch.nn.functional.interpolate(ds_up, size=x_ca.shape[-2:], mode='nearest')

        # --- Step 2: Spatial Fusion ---
        spatial_in = torch.cat([ds_up, x_ca], dim=1)
        spatial_in = self.act(self.spatial_bn(self.spatial_fusion_conv(spatial_in)))
        x_spatial = self.spatial_diffusion(spatial_in)

        # --- Step 3: Channel Fusion & Denoising ---
        channel_in = torch.cat([x_spatial, x_ca], dim=1)
        channel_in = self.act(self.channel_bn(self.channel_fusion_conv(channel_in)))
        x_full = self.channel_cleaner(channel_in)  # [B, C_curr, H, W]

        # --- Step 4: Selective Pruning (优胜劣汰) ---
        # 4.1 打分
        scores = self.ranker(x_full)  # [B, C, 1, 1]

        # 4.2 Batch 共识: 对 Batch 维度取平均，保证整个 Batch 切掉相同的通道
        global_scores = scores.mean(dim=0).flatten()  # [C]

        # 4.3 选出 Top-K 索引 (K = self.out_channels)
        _, topk_indices = torch.topk(global_scores, k=self.out_channels, dim=0)

        # 4.4 索引排序 (保持特征图原有语义顺序)
        topk_indices, _ = torch.sort(topk_indices)

        # 4.5 物理筛选 (Slicing)
        x_selected = torch.index_select(x_full, 1, topk_indices)

        return self.bn_final(x_selected)


class CEAttention(nn.Module):
    """
    [DAU 完整模块]
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1):
        super().__init__()
        if p is None: p = k // 2

        self.c1 = c1
        self.c2 = c2

        self.spatial = SpatialDiffusionBlock(c1, k=k, s=s, p=p)
        self.bn = nn.BatchNorm2d(c1)

        # 使用 Max 版 Task 模块
        self.task = TaskAwareBlock(c1, k=2)

        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        identity = x
        x = self.spatial(x)
        x = self.bn(x)
        x = self.task(x)
        if self.c1 == self.c2:
            x = x + identity
        return self.proj(x)


import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGate(nn.Module):
    """
    空间门控：生成 [B, 1, H, W] 的掩码
    """

    def __init__(self):
        super(SpatialGate, self).__init__()
        # 7x7 卷积感受野大，适合把破碎的小目标连成一片
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # [B, C, H, W] -> [B, 2, H, W]
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        scale = torch.cat([max_pool, avg_pool], dim=1)

        # [B, 1, H, W]
        return self.spatial(scale)


class VarianceChannelGate(nn.Module):
    """
    方差通道门控：基于空间掩码后的特征计算方差
    """

    def __init__(self, channels, reduction=8):
        super(VarianceChannelGate, self).__init__()
        # 守住底线 16，防止 Nano 模型通道被压得太扁
        mid_channels = max(channels // reduction, 16)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x, spatial_mask):
        b, c, h, w = x.shape

        # 1. 空间掩码聚焦
        x_focused = x * spatial_mask

        # 2. 计算方差 (Standard Deviation)
        # 纯背景通道 -> 方差低; 物体通道 -> 方差高
        # [B, C, H, W] -> [B, C, 1, 1]
        channel_std = torch.std(x_focused, dim=(2, 3), keepdim=True)

        # 3. 生成通道权重
        # MLP 输入: [B, C*1*1] -> Flatten -> [B, C]
        # MLP 输出: [B, C]
        channel_scale = self.mlp(channel_std)

        # [BUG FIX]: 必须 Reshape 回 [B, C, 1, 1] 才能和 [B, C, H, W] 相乘
        return channel_scale.view(b, c, 1, 1)


class VGASBlock(nn.Module):
    """
    [Variance-Guided Attention & Spatial Block]
    适用于 YOLO11n 的轻量级去噪模块
    """

    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k // 2

        # 1. 基础特征提取
        self.conv = nn.Conv2d(c1, c1, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

        # 2. 空间门控
        self.spatial_gate = SpatialGate()

        # 3. 方差通道门控
        self.var_channel_gate = VarianceChannelGate(c1, reduction=8)

        # 4. 投影层 (处理通道变化 c1 -> c2)
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

        # 5. 残差连接条件：必须 stride=1 且 输入输出通道一致
        self.add = s == 1 and c1 == c2

    def forward(self, x):
        # 基础卷积
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.act(feat)

        # Step 1: 空间掩码 [B, 1, H, W]
        s_mask = self.spatial_gate(feat)

        # Step 2: 方差通道权重 [B, C, 1, 1]
        c_scale = self.var_channel_gate(feat, s_mask)

        # Step 3: 双重加权 (空间去噪 + 通道筛选)
        # [B, C, H, W] * [B, 1, H, W] * [B, C, 1, 1] -> 广播机制生效
        feat = feat * s_mask * c_scale

        # 输出投影
        out = self.proj(feat)

        # 残差连接
        if self.add:
            return out + x
        else:
            return out


# ==========================================
# Sanity Check (自我体检)
# ==========================================
if __name__ == "__main__":
    # 测试 Case 1: 普通层 (Stride=1, Channel不变)
    x1 = torch.randn(2, 64, 80, 80)
    model1 = VGASBlock(64, 64, s=1)
    y1 = model1(x1)
    print(f"Test 1 (Normal): Input {x1.shape} -> Output {y1.shape}")

    # 测试 Case 2: 下采样层 (Stride=2, Channel翻倍)
    # 这就是之前报错 64 vs 32 的场景
    x2 = torch.randn(2, 64, 64, 64)
    model2 = VGASBlock(64, 128, s=2)
    y2 = model2(x2)
    print(f"Test 2 (Downsample): Input {x2.shape} -> Output {y2.shape}")

    # 梯度检查
    loss = y2.sum()
    loss.backward()
    print("Backward pass successful.")
class PolarizedAttention(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PolarizedAttention, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()
        input_x = input_x.view(batch, channel, height * width)
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)
        mask_ch = self.sigmoid(context)
        out = x * mask_ch
        return out

    def channel_pool(self, x):
        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)
        mask_sp = self.sigmoid(context)
        out = x * mask_sp
        return out

    def forward(self, x):
        # 并联
        # context_channel = self.spatial_pool(x)
        # context_spatial = self.channel_pool(x)
        # out = context_spatial + context_channel

        # 串联
        out = self.spatial_pool(x)
        out = self.channel_pool(out)

        return out


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = PolarizedAttention(512, 512).cuda()
    out = model(x)
    print(out.shape)


class EdgeSkipGate(nn.Module):
    """
    EdgeSkipGate: 单输入绿线门控残差模块（边缘引导，抑噪优先）

    输入/输出: (B, C, H, W) 不变

    设计：
    1) Channel Gate（SE-like）：抑制背景敏感通道
    2) Spatial Gate（edge-guided）：使用 [avg, max, edge_mag] 预测空间门控
       - edge_mag 由固定 Sobel 卷积提取（不增加可学习参数）
    3) ResNet-style：y = x + gamma * (x * gate)，gamma 初始为 0 保证稳定
    """

    def __init__(self, c1: int, k: int = 7, reduction: int = 16, eps: float = 1e-6):
        super().__init__()
        c1 = int(c1)
        k = int(k)
        reduction = int(reduction)
        self.eps = float(eps)

        # -------- Channel gate (SE) --------
        hidden = max(c1 // reduction, 4)
        self.cg_fc1 = nn.Conv2d(c1, hidden, 1, 1, 0, bias=True)
        self.cg_fc2 = nn.Conv2d(hidden, c1, 1, 1, 0, bias=True)
        self.act = nn.SiLU(inplace=False)

        # -------- Spatial gate (edge-guided) --------
        # input: [avg_map, max_map, edge_mag] -> 1 channel mask
        p = k // 2
        self.sg_conv = nn.Conv2d(3, 1, k, 1, p, bias=True)

        # -------- Fixed Sobel kernels (registered buffers) --------
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1,  2,  1],
                                [0,  0,  0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        # -------- Residual scale (ResNet-safe) --------
        self.gamma = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def _edge_mag(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算边缘幅值图 edge_mag: (B, 1, H, W)
        做法：先把多通道 x 压成 1 通道（均值），再用 Sobel 提取梯度
        """
        # x_gray: (B,1,H,W)
        x_gray = x.mean(dim=1, keepdim=True)

        # Sobel in FP32 for stability
        xg = x_gray.float()
        kx = self.sobel_x.to(device=x.device)
        ky = self.sobel_y.to(device=x.device)

        gx = F.conv2d(xg, kx, padding=1)
        gy = F.conv2d(xg, ky, padding=1)

        mag = torch.sqrt(gx * gx + gy * gy + self.eps)  # (B,1,H,W), FP32

        # 归一化到 [0,1]（按样本自适应，避免尺度漂）
        B = mag.shape[0]
        mag_flat = mag.view(B, -1)
        mag_min = mag_flat.min(dim=1)[0].view(B, 1, 1, 1)
        mag_max = mag_flat.max(dim=1)[0].view(B, 1, 1, 1)
        mag_n = (mag - mag_min) / (mag_max - mag_min + self.eps)

        return mag_n.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -------- Channel gate --------
        ch = F.adaptive_avg_pool2d(x, 1)
        ch = self.act(self.cg_fc1(ch))
        ch = torch.sigmoid(self.cg_fc2(ch))  # (B,C,1,1)

        # -------- Spatial gate (avg/max/edge) --------
        avg_map = x.mean(dim=1, keepdim=True)      # (B,1,H,W)
        max_map = x.amax(dim=1, keepdim=True)      # (B,1,H,W)
        edge_mag = self._edge_mag(x)               # (B,1,H,W)

        sp_in = torch.cat([avg_map, max_map, edge_mag], dim=1)  # (B,3,H,W)
        sp = torch.sigmoid(self.sg_conv(sp_in))                 # (B,1,H,W)

        gate = ch * sp  # broadcast -> (B,C,H,W)

        # -------- ResNet-style gated residual --------
        gamma = self.gamma.to(dtype=x.dtype)
        return x + gamma * (x * gate)

class HDRAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(HDRAB, self).__init__()
        kernel_size = 3
        reduction = 8
        reduction_2 = 2

        self.cab = CAB(in_channels, reduction, bias)

        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels // reduction_2, 1)

        self.conv1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size,
                               padding=1, dilation=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size,
                               padding=2, dilation=2, bias=bias)

        self.conv3 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size,
                               padding=3, dilation=3, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size,
                               padding=4, dilation=4, bias=bias)

        self.conv3_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size,
                                 padding=3, dilation=3, bias=bias)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size,
                                 padding=2, dilation=2, bias=bias)

        self.conv1_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size,
                                 padding=1, dilation=1, bias=bias)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv_tail = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size,
                                   padding=1, dilation=1, bias=bias)

        self.conv1x1_2 = nn.Conv2d(in_channels // reduction_2, in_channels, 1)

    def forward(self, y):
        y_d = self.conv1x1_1(y)
        y1 = self.conv1(y_d)
        y1_1 = self.relu1(y1)
        y2 = self.conv2(y1_1)
        y2_1 = y2 + y_d

        y3 = self.conv3(y2_1)
        y3_1 = self.relu3(y3)
        y4 = self.conv4(y3_1)
        y4_1 = y4 + y2_1

        y5 = self.conv3_1(y4_1)
        y5_1 = self.relu3_1(y5)
        y6 = self.conv2_1(y5_1 + y3)
        y6_1 = y6 + y4_1

        y7 = self.conv1_1(y6_1 + y2_1)
        y7_1 = self.relu1_1(y7)
        y8 = self.conv_tail(y7_1 + y1)
        y8_1 = y8 + y6_1

        y9 = self.cab(self.conv1x1_2(y8_1))
        y9_1 = y + y9

        return y9_1



class SPDConv(nn.Module):
    """标准卷积层，支持多种参数配置，包括输入通道数、输出通道数、卷积核大小、步幅、填充、分组、膨胀因子和激活函数。

    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int, optional): 卷积核大小，默认为1
        s (int, optional): 步幅，默认为1
        p (int or list, optional): 填充大小，默认为None
        g (int, optional): 分组数，默认为1
        d (int or list, optional): 膨胀因子，默认为1
        act (bool or nn.Module, optional): 是否使用激活函数，默认为True
    """
    default_act = nn.SiLU()  # 默认激活函数为SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """初始化卷积层。

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int, optional): 卷积核大小，默认为1
            s (int, optional): 步幅，默认为1
            p (int or list, optional): 填充大小，默认为None
            g (int, optional): 分组数，默认为1
            d (int or list, optional): 膨胀因子，默认为1
            act (bool or nn.Module, optional): 是否使用激活函数，默认为True
        """
        super().__init__()
        c1 = c1 * 4  # 将输入通道数乘以4
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # 定义卷积层
        self.bn = nn.BatchNorm2d(c2)  # 定义批量归一化层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()  # 定义激活函数

    def forward(self, x):
        """前向传播函数，对输入进行卷积、批量归一化和激活操作。

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 处理后的张量
        """
        # 将输入张量按通道维度进行切片并拼接，增加通道数
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        # 应用卷积、批量归一化和激活函数
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """前向传播函数（融合版本），对输入进行卷积和激活操作，不包含批量归一化。

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 处理后的张量
        """
        # 将输入张量按通道维度进行切片并拼接，增加通道数
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        # 应用卷积和激活函数
        return self.act(self.conv(x))


class CARAFE(nn.Module):
    """
    CARAFE 是一种上采样模块，通过学习的权重对特征图进行上采样。
    参数:
        c (int): 输入通道数
        k_enc (int): 编码器部分的卷积核大小
        k_up (int): 上采样时使用的 unfold 核大小
        c_mid (int): 中间通道数
        scale (int): 上采样倍率
    """

    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        super(CARAFE, self).__init__()
        print(k_enc,k_up)
        self.scale = scale  # 设置上采样倍率

        # 压缩输入通道到中间通道数
        self.comp = Conv(c, c_mid,act=nn.ReLU())

        # 编码器生成权重，输出通道数为(scale * k_up)^2
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)

        # 使用 PixelShuffle 进行上采样操作
        self.pix_shf = nn.PixelShuffle(scale)

        # 最近邻插值方法作为上采样操作
        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')

        # Unfold 操作提取感受野内的特征
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()  # 获取输入张量的形状
        h_, w_ = h * self.scale, w * self.scale  # 计算上采样后的高度和宽度

        W = self.comp(X)  # 压缩输入通道
        W = self.enc(W)  # 编码器生成权重
        W = self.pix_shf(W)  # Pixel Shuffle 上采样权重
        W = torch.softmax(W, dim=1)  # 对权重应用 softmax 归一化

        X = self.upsmp(X)  # 输入特征图使用最近邻插值上采样
        X = self.unfold(X)  # 提取上采样后特征的感受野
        X = X.view(b, c, -1, h_, w_)  # 调整视图以分离感受野维度

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # 应用注意力机制加权聚合
        return X  # 返回上采样后的特征图



class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


import torch
import torch.nn as nn
import torch.fft


class FCSA(nn.Module):
    """
    Frequency-coordinated Self-Attention (FCSA)
    针对小目标设计的频率协调自注意力模块。
    修复版：严格强制全过程频域计算使用 Float32，解决 cuFFT 在 FP16 下对非 2 幂次尺寸的限制。
    """

    def __init__(self, c1, c2):
        super().__init__()
        # 确保输入输出通道一致
        self.conv_wfc = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv_mfc = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv_fs_prime = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv_wfs = nn.Conv2d(c1, c1, 1, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cv_out = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # 记录原始精度 (通常是训练时的 Half/FP16)
        orig_dtype = x.dtype

        # ==========================================
        # 1. 频域通道调制 (Frequency-Guided Channel Modulation)
        # ==========================================
        # 强制转为 float32 以支持非 2 幂次尺寸 (如 160)
        x_fp32 = x.to(torch.float32)

        # F_FC = FFT(M)
        f_fc = torch.fft.fft2(x_fp32, norm='ortho')

        # W_FC 权重计算
        w_fc = self.conv_wfc(self.gap(x)).to(torch.float32)

        # M'_FC = IFFT(F_FC * W_FC)
        #
        m_fc_prime = torch.fft.ifft2(f_fc * w_fc, norm='ortho').real

        # 转回原始精度进行空域混合
        m_fc_prime = m_fc_prime.to(orig_dtype)
        m_fc = m_fc_prime * self.conv_mfc(self.gap(m_fc_prime))

        # ==========================================
        # 2. 频域空间调制 (Frequency-Guided Spatial Modulation)
        # ==========================================
        # 【关键修复点】：确保进入 conv 后、进入 fft 前，全部转换为 fp32
        m_fc_fp32 = m_fc.to(torch.float32)

        # M'_FS = FFT(Conv1x1(M_FC))
        # 先执行卷积，再强制转 fp32，最后进 fft
        f_fs_input = self.conv_fs_prime(m_fc).to(torch.float32)
        f_fs_prime = torch.fft.fft2(f_fs_input, norm='ortho')

        # W_FS = Conv1x1(M_FC)
        w_fs = self.conv_wfs(m_fc).to(torch.float32)

        # M_FS = IFFT(M'_FS * W_FS)
        #
        m_fs = torch.fft.ifft2(f_fs_prime * w_fs, norm='ortho').real

        # 最终输出转回原始精度并处理通道
        return self.cv_out(m_fs.to(orig_dtype))

import torch
import torch.nn as nn
from einops import rearrange
from .conv import Conv


class LowRankMix(nn.Module):
    """
    低秩 1x1 混合层
    用 C -> r -> C 替代原始 C -> C 的重型 1x1
    并保留残差，避免直接伤主表达
    """
    def __init__(self, dim, ratio=0.25, min_dim=64):
        super().__init__()
        hidden = max(int(dim * ratio), min_dim)

        self.down = nn.Conv2d(dim, hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)

        self.up = nn.Conv2d(hidden, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

        # 小尺度残差系数，初始更稳
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        out = self.down(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.up(out)
        out = self.bn2(out)

        return x + self.beta * out


class SADEConv(nn.Module):
    """
    只改 feature 分支第二个超重 1x1：
    普通 1x1 -> 低秩残差混合
    """
    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        weight_act="softmax",
        temperature=1.0,
        mix_ratio=0.25,   # 新增：低秩比例
    ):
        super().__init__()
        self.kernel_size = k
        self.stride = s
        self.k2 = k ** 2
        self.weight_act = weight_act
        self.temperature = temperature

        # -------------------------
        # Path1: attention weight branch
        # 原样保留
        # -------------------------
        self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=s, padding=k // 2)
        self.max_pool = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

        self.weight_dw3 = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        self.weight_dw5 = nn.Conv2d(c1, c1, 5, 1, 2, groups=c1, bias=False)
        self.weight_dwd = nn.Conv2d(c1, c1, 3, 1, 2, dilation=2, groups=c1, bias=False)

        self.weight_fuse = nn.Conv2d(
            c1 * 3,
            c1 * self.k2,
            kernel_size=1,
            groups=c1,
            bias=False
        )

        # -------------------------
        # Path2: RF feature branch
        # 第一层保留
        # 第二层重构为低秩混合
        # -------------------------
        feat_dim = c1 * self.k2

        self.feature_pre = nn.Sequential(
            nn.Conv2d(
                c1,
                feat_dim,
                kernel_size=k,
                stride=s,
                padding=k // 2,
                groups=c1,
                bias=False
            ),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        self.feature_mix = LowRankMix(feat_dim, ratio=mix_ratio, min_dim=64)

        self.feature_post = nn.ReLU(inplace=True)

        # -------------------------
        # Attention output branch
        # -------------------------
        self.att_conv = Conv(c1, c2, k=k, s=k, p=0)

        # -------------------------
        # Base residual branch
        # -------------------------
        self.base_proj = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(c2)
        )

        # -------------------------
        # Gate branch
        # 保持你原逻辑
        # -------------------------
        self.gate = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def _get_attention_weight(self, x):
        b, c = x.shape[:2]

        pooled = self.avg_pool(x) + self.max_pool(x)

        w3 = self.weight_dw3(pooled)
        w5 = self.weight_dw5(pooled)
        wd = self.weight_dwd(pooled)

        weight = torch.cat([w3, w5, wd], dim=1)
        weight = self.weight_fuse(weight)

        h, w = weight.shape[2:]
        weight = weight.view(b, c, self.k2, h, w)

        if self.weight_act == "softmax":
            weight = torch.softmax(weight / self.temperature, dim=2)
        elif self.weight_act == "sigmoid":
            weight = torch.sigmoid(weight)
        else:
            raise ValueError(f"Unsupported weight_act: {self.weight_act}")

        return weight

    def forward(self, x):
        b, c = x.shape[:2]

        # Path1
        weight = self._get_attention_weight(x)   # [B, C, k^2, h, w]
        h, w = weight.shape[3:]

        # Path2
        feature = self.feature_pre(x)
        feature = self.feature_mix(feature)
        feature = self.feature_post(feature)
        feature = feature.view(b, c, self.k2, h, w)

        # Attention aggregation
        weighted_data = feature * weight
        conv_data = rearrange(
            weighted_data,
            "b c (n1 n2) h w -> b c (h n1) (w n2)",
            n1=self.kernel_size,
            n2=self.kernel_size
        )
        x_att = self.att_conv(conv_data)

        # Base residual
        x_base = self.base_proj(x)

        # Gate
        alpha = self.gate(x_base)

        # Final fusion
        y = x_base + alpha * x_att
        return y


# Backward-compatible alias for older model YAML files.
MSRA_RFAConv = SADEConv


class ESADConv(nn.Module):
    """
    Edge-aware spatial adaptive downsampling convolution.

    The module preserves fine spatial detail with a space-to-depth path, enhances
    directional high-frequency responses, and fuses base/scale/detail experts with
    spatially adaptive weights.
    """

    def __init__(self, c1, c2, k=3, s=2, gate_ratio=0.25):
        super().__init__()
        self.stride = s

        hidden = max(int(c2 * gate_ratio), 16)

        self.base_branch = Conv(c1, c2, k=k, s=s)
        self.spd_branch = Conv(c1 * 4 if s == 2 else c1, c2, k=1, s=1)

        self.edge_h = nn.Conv2d(c1, c1, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=c1, bias=False)
        self.edge_v = nn.Conv2d(c1, c1, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=c1, bias=False)
        self.edge_d = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=2, dilation=2, groups=c1, bias=False)
        self.edge_fuse = Conv(c1 * 3, c2, k=1, s=1)

        self.scale_3 = nn.Conv2d(c1, c1, kernel_size=3, stride=s, padding=1, groups=c1, bias=False)
        self.scale_5 = nn.Conv2d(c1, c1, kernel_size=5, stride=s, padding=2, groups=c1, bias=False)
        self.scale_d = nn.Conv2d(c1, c1, kernel_size=3, stride=s, padding=2, dilation=2, groups=c1, bias=False)
        self.scale_fuse = Conv(c1 * 3, c2, k=1, s=1)

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(c2 * 4, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=1, bias=True),
        )
        self.out = Conv(c2, c2, k=1, s=1)

    @staticmethod
    def _match_size(x, size):
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="nearest")

    def _downsample_detail(self, x, size):
        detail = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        if self.stride > 1:
            detail = F.avg_pool2d(detail, kernel_size=self.stride, stride=self.stride, ceil_mode=True)
        return self._match_size(detail, size)

    def _space_to_depth(self, x, size):
        if self.stride == 2:
            h, w = x.shape[-2:]
            pad_h = h % 2
            pad_w = w % 2
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))
            x = F.pixel_unshuffle(x, 2)
        elif self.stride > 1:
            x = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride, ceil_mode=True)
        return self._match_size(x, size)

    def forward(self, x):
        base = self.base_branch(x)
        size = base.shape[-2:]

        spd = self.spd_branch(self._space_to_depth(x, size))

        detail = self._downsample_detail(x, size)
        edge = self.edge_fuse(torch.cat((self.edge_h(detail), self.edge_v(detail), self.edge_d(detail)), dim=1))

        scale = self.scale_fuse(torch.cat((self.scale_3(x), self.scale_5(x), self.scale_d(x)), dim=1))
        scale = self._match_size(scale, size)

        experts = torch.stack((base, spd, edge, scale), dim=1)
        gate = torch.softmax(self.fusion_gate(torch.cat((base, spd, edge, scale), dim=1)), dim=1)
        y = (experts * gate.unsqueeze(2)).sum(dim=1)
        return self.out(y)


# ===== add into ultralytics/nn/modules/block.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D(nn.Module):
    """
    Simple DropBlock for 2D feature maps.
    Applied only during training.
    """
    def __init__(self, drop_prob=0.0, block_size=3):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.block_size = int(block_size)

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x

        n, c, h, w = x.shape
        if h < self.block_size or w < self.block_size:
            return x

        gamma = self._compute_gamma(x)

        mask = (torch.rand(n, c, h, w, device=x.device, dtype=x.dtype) < gamma).float()
        block_mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask
        normalize_scale = block_mask.numel() / (block_mask.sum() + 1e-6)
        return x * block_mask * normalize_scale

    def _compute_gamma(self, x):
        _, _, h, w = x.shape
        valid_h = h - self.block_size + 1
        valid_w = w - self.block_size + 1
        return self.drop_prob * (h * w) / (self.block_size ** 2) / (valid_h * valid_w + 1e-6)


class _NASConvBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        p = (k - 1) // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))


class _NASReLUConvBN(nn.Module):
    """
    ReLU -> Conv -> BN -> DropBlock
    """
    def __init__(self, c1, c2, k=3, s=1, p=1, drop_prob=0.0, block_size=3):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.drop = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

    def forward(self, x):
        x = self.act(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.drop(x)
        return x


class _NASBaseMergeCell(nn.Module):
    def __init__(self, channels, with_out_conv=True, drop_prob=0.0, block_size=3):
        super().__init__()
        self.with_out_conv = with_out_conv
        self.out_conv = (
            _NASReLUConvBN(channels, channels, 3, 1, 1, drop_prob=drop_prob, block_size=block_size)
            if with_out_conv else nn.Identity()
        )

    @staticmethod
    def _resize(x, out_size):
        if x.shape[-2:] == out_size:
            return x

        h, w = x.shape[-2:]
        oh, ow = out_size

        if h < oh or w < ow:
            return F.interpolate(x, size=out_size, mode="nearest")

        kh = max(h // oh, 1)
        kw = max(w // ow, 1)
        return F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))

    def _binary_op(self, x1, x2):
        raise NotImplementedError

    def forward(self, x1, x2, out_size):
        x1 = self._resize(x1, out_size)
        x2 = self._resize(x2, out_size)
        x = self._binary_op(x1, x2)
        return self.out_conv(x)


class _NASSumCell(_NASBaseMergeCell):
    def __init__(self, channels, with_out_conv=True, drop_prob=0.0, block_size=3):
        super().__init__(channels, with_out_conv, drop_prob=drop_prob, block_size=block_size)

    def _binary_op(self, x1, x2):
        return x1 + x2


class _NASGlobalPoolingCell(_NASBaseMergeCell):
    def __init__(self, channels, with_out_conv=True, drop_prob=0.0, block_size=3):
        super().__init__(channels, with_out_conv, drop_prob=drop_prob, block_size=block_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _binary_op(self, x1, x2):
        att = self.pool(x2).sigmoid()
        return x2 + att * x1


class NASFPN(nn.Module):
    """
    Modified NAS-FPN for Ultralytics.
    Input:  [C2, C3, C4, C5]
    Output: [P2, P3, P4, P5, P6]
    Usually Detect uses P2, P3, P4 (or P2, P3, P4, P5).
    """

    def __init__(
        self,
        in_channels,
        out_channels=64,
        num_outs=5,
        stack_times=1,
        drop_prob=0.05,
        block_size=3,
    ):
        super().__init__()
        assert isinstance(in_channels, (list, tuple)), "in_channels must be a list/tuple"
        assert len(in_channels) == 4, "NASFPN expects 4 input feature levels: [C2, C3, C4, C5]"

        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.stack_times = stack_times
        self.drop_prob = drop_prob
        self.block_size = block_size

        # lateral 1x1
        self.lateral_convs = nn.ModuleList([_NASConvBN(c, out_channels, 1, 1, 0) for c in self.in_channels])

        # build extra level: P6 from P5
        self.extra_downsamples = nn.ModuleList()
        extra_levels = num_outs - len(in_channels)  # normally 1
        for _ in range(extra_levels):
            self.extra_downsamples.append(
                nn.Sequential(
                    _NASConvBN(out_channels, out_channels, 1, 1, 0),
                    nn.MaxPool2d(2, 2)
                )
            )

        # shifted topology: original P3-P7 -> now P2-P6
        self.fpn_stages = nn.ModuleList()
        for _ in range(stack_times):
            stage = nn.ModuleDict({
                "gp_53_3": _NASGlobalPoolingCell(
                    out_channels, with_out_conv=True, drop_prob=drop_prob, block_size=block_size
                ),   # gp(p5, p3) -> p3_1
                "sum_33_3": _NASSumCell(
                    out_channels, with_out_conv=True, drop_prob=drop_prob, block_size=block_size
                ),    # sum(p3_1, p3) -> p3_2
                "sum_32_2": _NASSumCell(
                    out_channels, with_out_conv=True, drop_prob=drop_prob, block_size=block_size
                ),    # sum(p3_2, p2) -> p2_out
                "sum_23_3": _NASSumCell(
                    out_channels, with_out_conv=True, drop_prob=drop_prob, block_size=block_size
                ),    # sum(p2_out, p3_2) -> p3_out
                "gp_32_4": _NASGlobalPoolingCell(
                    out_channels, with_out_conv=False, drop_prob=drop_prob, block_size=block_size
                ),   # gp(p3_out, p2_out) -> p4_tmp
                "sum_44_4": _NASSumCell(
                    out_channels, with_out_conv=True, drop_prob=drop_prob, block_size=block_size
                ),    # sum(p4, p4_tmp) -> p4_out
                "gp_43_6": _NASGlobalPoolingCell(
                    out_channels, with_out_conv=False, drop_prob=drop_prob, block_size=block_size
                ),   # gp(p4_out, p3_2) -> p6_tmp
                "sum_66_6": _NASSumCell(
                    out_channels, with_out_conv=True, drop_prob=drop_prob, block_size=block_size
                ),    # sum(p6, p6_tmp) -> p6_out
                "gp_64_5": _NASGlobalPoolingCell(
                    out_channels, with_out_conv=True, drop_prob=drop_prob, block_size=block_size
                ),   # gp(p6_out, p4_out) -> p5_out
            })
            self.fpn_stages.append(stage)

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple)), "NASFPN forward expects a list/tuple of feature maps"
        assert len(inputs) == 4, "NASFPN forward expects [C2, C3, C4, C5]"

        feats = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        while len(feats) < self.num_outs:
            feats.append(self.extra_downsamples[len(feats) - len(self.in_channels)](feats[-1]))

        p2, p3, p4, p5, p6 = feats[:5]

        for stage in self.fpn_stages:
            p3_1 = stage["gp_53_3"](p5, p3, out_size=p3.shape[-2:])
            p3_2 = stage["sum_33_3"](p3_1, p3, out_size=p3.shape[-2:])
            p2   = stage["sum_32_2"](p3_2, p2, out_size=p2.shape[-2:])
            p3   = stage["sum_23_3"](p2, p3_2, out_size=p3.shape[-2:])
            p4_t = stage["gp_32_4"](p3, p2, out_size=p4.shape[-2:])
            p4   = stage["sum_44_4"](p4, p4_t, out_size=p4.shape[-2:])
            p6_t = stage["gp_43_6"](p4, p3_2, out_size=p6.shape[-2:])
            p6   = stage["sum_66_6"](p6, p6_t, out_size=p6.shape[-2:])
            p5   = stage["gp_64_5"](p6, p4, out_size=p5.shape[-2:])

        return [p2, p3, p4, p5, p6]



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths per sample."""
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths per sample."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: Tensor with shape (B, H, W, C).
        window_size: Window size.

    Returns:
        windows: Tensor with shape (num_windows * B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Reverse windows back to feature map.

    Args:
        windows: Tensor with shape (num_windows * B, window_size, window_size, C).
        window_size: Window size.
        H: Feature height.
        W: Feature width.

    Returns:
        x: Tensor with shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """MLP used in Swin Transformer."""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads
            )
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(
            B_,
            N,
            3,
            self.num_heads,
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW,
                nW,
                self.num_heads,
                N,
                N
            ) + mask.unsqueeze(1).unsqueeze(0)

            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerLayer(nn.Module):
    """Swin Transformer layer used in C3STR."""
    def __init__(
        self,
        c,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        if num_heads > 10:
            drop_path = 0.1

        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(c)
        self.attn = WindowAttention(
            c,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(c)

        mlp_hidden_dim = int(c * mlp_ratio)
        self.mlp = Mlp(
            in_features=c,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def create_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)

        h_slices = (
            (0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0,
            torch.tensor(-100.0, device=x.device)
        ).masked_fill(
            attn_mask == 0,
            torch.tensor(0.0, device=x.device)
        )

        return attn_mask

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()

        attn_mask = self.create_mask(x, h, w)

        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)
            )
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(
            -1,
            self.window_size,
            self.window_size,
            c
        )

        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block."""
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()

        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        self.window_size = window_size
        self.shift_size = window_size // 2

        self.tr = nn.Sequential(*(
            SwinTransformerLayer(
                c2,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else self.shift_size
            )
            for i in range(num_layers)
        ))

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)

        x = self.tr(x)
        return x


class C3STR(C3):
    """C3 module with Swin Transformer Block."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        c_ = int(c2 * e)
        num_heads = max(1, c_ // 32)

        self.m = SwinTransformerBlock(
            c_,
            c_,
            num_heads=num_heads,
            num_layers=n
        )
