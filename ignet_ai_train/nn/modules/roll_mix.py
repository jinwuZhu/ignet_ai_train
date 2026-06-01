import torch
import torch.nn as nn

class Roll8(nn.Module):
    """
    8方向滚动模块，输入 B, C, H, W  输出 B, 8*C, H, W
    Args:
        step (int, optional): 滚动步长，默认为1
    """
    def __init__(self, step:int = 1):
        """
        初始化函数。
        """
        super(Roll8, self).__init__()
        self.roll_dirs = [
            (-1 * step, 0),  # 上
            (1 * step, 0),   # 下
            (0, -1 * step),  # 左
            (0, 1 * step),   # 右
            (-1 * step, -1 * step), # 左上
            (-1 * step, 1 * step),  # 右上
            (1 * step, -1 * step),  # 左下
            (1 * step, 1 * step),   # 右下
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。

        Returns:
            torch.Tensor: 输出张量，形状为 (B, 8*C, H, W)。
        """
        # 定义8个方向的偏移量 (height_shift, width_shift)
        
        # 将所有8个方向的滚动结果堆叠起来，然后在维度0上求均值
        rolled_tensors = [] # list B, C, H, W
        for h_shift, w_shift in self.roll_dirs:
            # torch.roll 循环移位
            rolled = torch.roll(x, shifts=(h_shift, w_shift), dims=(2, 3))
            rolled_tensors.append(rolled)
        
        # list of B, C, H, W -> B, 8*C + 1, H, W
        mixed_feature = torch.cat(rolled_tensors, dim=1)
        return mixed_feature

    
class RollMix8(nn.Module):
    """
    8方位滚动混合模块，将一个输入张量按照8个方向进行滚动相加并取均值， 输入 B,C,H,W, 输出 B,C,H,W
    """
    def __init__(self,step:int = 1):
        """
        初始化函数。

        Args:
            channels (int): 输入张量的通道数。
        """
        super(RollMix8, self).__init__()
        self.roll_dirs = [
            (-1 * step, 0),  # 上
            (1 * step, 0),   # 下
            (0, -1 * step),  # 左
            (0, 1 * step),   # 右
            (-1 * step, -1 * step), # 左上
            (-1 * step, 1 * step),  # 右上
            (1 * step, -1 * step),  # 左下
            (1 * step, 1 * step),   # 右下
        ]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。

        Returns:
            torch.Tensor: 输出张量，形状为 (B, C, H, W)。
        """
        
        # 将所有8个方向的滚动结果堆叠起来，然后在维度0上求均值
        rolled_tensors = []
        for h_shift, w_shift in self.roll_dirs:
            # torch.roll 循环移位
            rolled = torch.roll(x, shifts=(h_shift, w_shift), dims=(2, 3))
            rolled_tensors.append(rolled)
        
        # 计算8个滚动结果的平均值
        # shape: (8, B, C, H, W) -> (B, C, H, W)
        mixed_feature = torch.stack(rolled_tensors).mean(dim=0)

        out = x + mixed_feature
        
        return out

# --- 示例用法 ---
if __name__ == "__main__":
    module = Roll8()
    
    # 创建一个模拟输入张量 (Batch=2, Channel=3, Height=4, Width=5)
    input_tensor = torch.randn(2, 3, 4, 5)
    
    print(f"输入形状: {input_tensor.shape}")
    
    # 执行前向传播
    output_tensor = module(input_tensor)
    
    print(f"输出形状: {output_tensor.shape}")
    # 预期输出形状: torch.Size([2, 3, 4, 5])