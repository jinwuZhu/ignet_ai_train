import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignet_ai_train.nn.modules import CReLU
from torchsummary import summary
import cv2

from ignet_ai_train.model_builder import build_model_by_yaml
class SuperResolutionX4(nn.Module):
    def __init__(self,c:int = 3):
        super().__init__()
        self.backbone:nn.Module = build_model_by_yaml("test/models/super_resolution_x2s.yaml",c=c)
    
    def forward(self,x):
        x2 = self.backbone(x)
        x4 = self.backbone(x2)
        return x4

if __name__ == "__main__":
    # model = SuperResolutionX4()
    model:nn.Module = build_model_by_yaml("test/models/super_resolution_x2s.yaml",c=3)
    pkt = torch.load("D:/tmp/checkpoint_GAN_9.pth",map_location="cpu")
    model.load_state_dict(pkt["model_state_dict"])
    model.eval()

    x = cv2.imread("cache/test.png",cv2.IMREAD_COLOR)
    x =  x.astype(np.float32) / 255.0
    # normalize
    mean=0.5
    std = 0.5
    x = (x - mean) / std
    x:torch.FloatTensor = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float()
    
    y:torch.FloatTensor = model(x)
    # export onnx
    print(f"输出形状: {y.shape}") # B,C,H,W std=0.5,mean=0.5
    # to image
    image:np.ndarray = y[0].squeeze().cpu().detach().numpy() 
    image = (image * std + mean)# 反归一化
    image = np.clip(image,0,1) # 确保像素值在[0, 1]范围内
    image = ( image * 255.0).astype(np.uint8) # 转换为uint8类型 
    # 改变形状以匹配OpenCV的期望输入
    image = image.transpose(1, 2, 0)
    cv2.imwrite("cache/test_out.jpg",image)
    # model.half()
    # x = x.half()
    torch.onnx.export(
        model, 
        x, 
        "checkpoints/super_resolution_x2s.onnx", 
        export_params=True,
        external_data=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: 'batch', 2: 'height', 3: 'width'},
            "output": {0: 'batch', 2: 'height', 3: 'width'}
        })