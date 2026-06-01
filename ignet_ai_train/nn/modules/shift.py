import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class Shift8(nn.Module):
    def __init__(self,groups:int=4):
        """
        https://arxiv.org/abs/2307.16140
        """
        
        super().__init__()
        self.g = groups
        self.shifts_list = [
            ( 1, 0), # 上
            (-1, 0), # 下
            ( 0, 1), # 左
            ( 0,-1), # 右
            ( 1, 1), # 上左
            ( 1,-1), # 上右
            (-1, 1), # 下左
            (-1,-1)  # 下右
        ]
        
    def forward(self,x):
        _,c,_,_ = x.shape

        assert c  == self.g * 8
        
        #
        x[:, 0 : 1 * self.g, :, :] = torch.roll(x[:, 0 : 1 * self.g, :, :], shifts=self.shifts_list[0], dims=(2, 3))
        x[:, 1 * self.g : 2 * self.g, :, :] = torch.roll(x[:, 1 * self.g : 2 * self.g, :, :], shifts=self.shifts_list[1], dims=(2, 3))
        x[:, 2 * self.g : 3 * self.g, :, :] = torch.roll(x[:, 2 * self.g : 3 * self.g, :, :], shifts=self.shifts_list[2], dims=(2, 3))
        x[:, 3 * self.g : 4 * self.g, :, :] = torch.roll(x[:, 3 * self.g : 4 * self.g, :, :], shifts=self.shifts_list[3], dims=(2, 3))
        x[:, 4 * self.g : 5 * self.g, :, :] = torch.roll(x[:, 4 * self.g : 5 * self.g, :, :], shifts=self.shifts_list[4], dims=(2, 3))
        x[:, 5 * self.g : 6 * self.g, :, :] = torch.roll(x[:, 5 * self.g : 6 * self.g, :, :], shifts=self.shifts_list[5], dims=(2, 3))
        x[:, 6 * self.g : 7 * self.g, :, :] = torch.roll(x[:, 6 * self.g : 7 * self.g, :, :], shifts=self.shifts_list[6], dims=(2, 3))
        x[:, 7 * self.g : 8 * self.g, :, :] = torch.roll(x[:, 7 * self.g : 8 * self.g, :, :], shifts=self.shifts_list[7], dims=(2, 3))
        return x
    
class ResidualBlockShift(nn.Module):
    def __init__(
            self,
            num_feat:int=64, 
            scale:float = 1.0, 
            sigmoid:Literal['sigmoid','relu'] = 'sigmoid'):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_feat,num_feat,kernel_size=1)
        self.shift = Shift8(groups=num_feat//8)
        self.sigmoid = nn.Sigmoid() if sigmoid == 'sigmoid' else nn.ReLU(True)
        self.conv2 = nn.Conv2d(num_feat,num_feat,kernel_size=1)
    
    def forward(self,x):
        identity = x
        out = self.conv2(self.sigmoid(self.shift(self.conv1(x))))
        return identity + out * self.scale

if __name__ == "__main__":
    x = torch.randn(1,64,64,64)
    model = ResidualBlockShift()
    out = model(x)
    torch.mean(out,dim=(2,3))
    print(out.shape)