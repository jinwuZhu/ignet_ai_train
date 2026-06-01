import torch
import torch.nn as nn

class CReLU(nn.Module):
    def __init__(self,inplace:bool=True):
        super(CReLU, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x_neg = -x                   # Neg
        x = torch.cat([x, x_neg], dim=1)  # Concat (channel翻倍)
        x = self.relu(x)             # ReLU
        return x
