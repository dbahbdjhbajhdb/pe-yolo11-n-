import math
import torch
import torch.nn as nn

from .conv import Conv, autopad


# ============================================================
# IMPORTANT:
# The uploaded YOLO-KAN module code depends on a class/function
# named KAN, but the definition of KAN is NOT included in the
# provided text.
#
# You must either:
#   1. Copy the official KAN class into this file, above KAN_Block;
# or
#   2. Import it from the correct file in the YOLO-KAN repository.
#
# Example:
# from .kan_layer import KAN
# ============================================================


class DepthwiseFlattenLayer(nn.Module):
    """Flatten layer using depthwise convolution."""

    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=1,
            groups=in_chans
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H, W]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x = self.norm(x)
        return x, H, W


class MaxPoolFlattenLayer(nn.Module):
    """Flatten layer using max pooling + 1x1 convolution."""

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_chans)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.norm(x)

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class MaxPoolFlattenLayer2(nn.Module):
    """Max pooling + 1x1 convolution without sequence flattening."""

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_chans)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.norm(x)
        return x


class FlattenLayer(nn.Module):
    """Image-to-patch embedding layer for KAN-based blocks."""

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H, W]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x = self.norm(x)
        return x, H, W


class KAN_Block(nn.Module):
    """KAN block with KAN layer, depthwise convolution, BN and SiLU activation."""

    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
        patch_size=1,
        stride=1
    ):
        super().__init__()

        self.flatten = DepthwiseFlattenLayer(c1, c2)

        # NOTE:
        # KAN must be defined or imported before using this class.
        self.kanlayer = KAN(
            [c1, c2],
            grid_size=4,
            spline_order=3,
            scale_noise=0.1
        )

        self.dwconv = nn.Conv2d(
            c2,
            c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            nn.SiLU()
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        flattened_x, H, W = self.flatten(x)  # [B, N, embed_dim]

        x = self.kanlayer(flattened_x)  # [B, N, c2]

        # [B, N, c2] -> [B, c2, H, W]
        x = x.transpose(1, 2).contiguous().view(
            x.shape[0],
            x.shape[-1],
            H,
            W
        )

        x = self.dwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class KAN_Block2(nn.Module):
    """Alternative KAN block using direct pixel-wise flattening."""

    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True
    ):
        super().__init__()

        self.kanlayer = KAN(
            [c1, c2],
            grid_size=4,
            spline_order=3,
            scale_noise=0.1
        )

        self.dwconv = nn.Conv2d(
            c2,
            c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            nn.SiLU()
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = self.kanlayer(x)

        x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.dwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class Bottleneck_KAN_1D(nn.Module):
    """Bottleneck block using Conv + KAN_Block."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, k[0])
        self.cv2 = KAN_Block(c_, c2, k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv1(x)
        out = self.cv2(out)

        return x + out if self.add else out


class C3_KAN(nn.Module):
    """CSP bottleneck with KAN-enhanced bottleneck blocks."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
        k=3
    ):
        super().__init__()

        self.c_ = int(c2 * e)

        self.cv1 = Conv(c1, self.c_, 1, 1)
        self.cv2 = Conv(c1, self.c_, 1, 1)
        self.cv3 = Conv(2 * self.c_, c2, 1)

        self.flatten = FlattenLayer(
            patch_size=5,
            stride=1,
            in_chans=self.c_,
            embed_dim=self.c_
        )

        self.m = nn.Sequential(
            *(
                Bottleneck_KAN_1D(
                    self.c_,
                    self.c_,
                    shortcut,
                    g,
                    k=(k, k),
                    e=1.0
                )
                for _ in range(n)
            )
        )

    def forward(self, x):
        y1 = self.cv1(x)

        flattened_y1, H, W = self.flatten(y1)

        y1 = flattened_y1.transpose(1, 2).contiguous().view(
            y1.shape[0],
            self.c_,
            H,
            W
        )

        y1 = self.m(y1)

        y2 = self.cv2(x)

        return self.cv3(torch.cat((y1, y2), dim=1))


class C2f_KAN_1D(nn.Module):
    """C2f module with KAN-enhanced bottleneck blocks."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5
    ):
        super().__init__()

        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            Bottleneck_KAN_1D(
                self.c,
                self.c,
                shortcut,
                g,
                k=(3, 3),
                e=1.0
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))

        x = y[-1]
        for m in self.m:
            x = m(x)

        y.append(x)

        return self.cv2(torch.cat(y, dim=1))


class C2f_KAN_1D2(nn.Module):
    """C2f module with flattening before KAN bottleneck blocks."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5
    ):
        super().__init__()

        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.flatten = FlattenLayer(
            patch_size=7,
            stride=1,
            in_chans=self.c,
            embed_dim=self.c
        )

        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            Bottleneck_KAN_1D(
                self.c,
                self.c,
                shortcut,
                g,
                k=(3, 3),
                e=1.0
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))

        flattened_y, H, W = self.flatten(y[-1])

        x = flattened_y.transpose(1, 2).contiguous().view(
            y[-1].shape[0],
            self.c,
            H,
            W
        )

        for m in self.m:
            x = m(x)

        y.append(x)

        return self.cv2(torch.cat(y, dim=1))


class C2f_KAN(nn.Module):
    """C2f module using nested C3_KAN blocks."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5
    ):
        super().__init__()

        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.flatten = FlattenLayer(
            patch_size=5,
            stride=1,
            in_chans=self.c,
            embed_dim=self.c
        )

        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            C3_KAN(
                self.c,
                self.c,
                n=2,
                shortcut=shortcut,
                g=g
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))

        flattened_y, H, W = self.flatten(y[-1])

        x = flattened_y.transpose(1, 2).contiguous().view(
            y[-1].shape[0],
            self.c,
            H,
            W
        )

        for m in self.m:
            x = m(x)

        y.append(x)

        return self.cv2(torch.cat(y, dim=1))