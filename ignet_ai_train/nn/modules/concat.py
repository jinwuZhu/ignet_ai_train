import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class Repeat(nn.Module):
    def __init__(self,repeats:Sequence[int]):
        super().__init__()
        self.repeats = repeats
    
    def forward(self,x:torch.Tensor):
        return x.repeat(*self.repeats)

class Concat(nn.Module):
    """
    Concatenate two tensors
    """
    def __init__(self,dim:int = 1):
        super().__init__()
        self.dim = dim
    def forward(self, *args):
        return torch.cat(args,dim=self.dim)



class SimpleAddFusion(nn.Module):
    """
    简单的特征相加融合模块。
    仅适用于两个输入特征具有相同通道数的情况。
    """
    def __init__(self):
        super(SimpleAddFusion, self).__init__()

    def forward(self, feat1, feat2):
        if feat1.shape != feat2.shape:
            raise ValueError("输入特征的形状必须完全相同才能进行相加。")
        return feat1 + feat2

class DepthwiseSeparableFusion(nn.Module):
    """
    使用深度可分离卷积进行特征融合，以降低计算复杂度。
    """
    def __init__(self, in_channels_1, in_channels_2, out_channels=None):
        super(DepthwiseSeparableFusion, self).__init__()
        total_in_channels = in_channels_1 + in_channels_2
        self.out_channels = out_channels or max(in_channels_1, in_channels_2)
        
        # 深度卷积 (Depthwise Conv)
        self.depthwise = nn.Conv2d(total_in_channels, total_in_channels, 
                                   kernel_size=3, padding=1, groups=total_in_channels, bias=False)
        # 逐点卷积 (Pointwise Conv)
        self.pointwise = nn.Conv2d(total_in_channels, self.out_channels, 
                                   kernel_size=1, bias=False)

    def forward(self, feat1, feat2):
        if feat1.shape[2:] != feat2.shape[2:]:
            # 如果空间尺寸不同，可以插值调整到一致
            if feat1.shape[2] < feat2.shape[2]:
                feat1 = F.interpolate(feat1, size=feat2.shape[2:], mode='bilinear', align_corners=False)
            else:
                feat2 = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([feat1, feat2], dim=1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SEFusion(nn.Module):
    """
    结合了特征拼接和SE注意力机制的融合模块。
    """
    def __init__(self, in_channels_1, in_channels_2, reduction=16):
        super(SEFusion, self).__init__()
        total_channels = in_channels_1 + in_channels_2
        self.out_channels = max(in_channels_1, in_channels_2)
        
        # 用于融合的1x1卷积
        self.conv_fuse = nn.Conv2d(total_channels, self.out_channels, kernel_size=1)
        
        # SE 注意力模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels, self.out_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels // reduction, self.out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x = self.conv_fuse(x)
        attention_weights = self.se(x)
        output = x * attention_weights
        return output


if __name__ == "__main__":
    N, H, W = 2, 14, 14
    feat_e = torch.randn(N, 64, H, W)
    feat_f = torch.randn(N, 64, H, W)

    se_fusion = SEFusion(in_channels_1=64, in_channels_2=64)
    output = se_fusion(feat_e, feat_f)

    print(f"SE注意力融合后输出的形状: {output.shape}") # [2, 128, 32, 32]