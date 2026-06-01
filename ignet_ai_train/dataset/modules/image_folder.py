import os
from PIL import Image

from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision.transforms import transforms

from ignet_ai_train.utils.download import auto_cache_download_list


def create_image_transforms(
        resize:int|tuple[int,int]= None,
        crop_size:int|tuple[int,...] = 256, 
        crop_model:str = 'center', 
        color_space:str='rgb', 
        mean:float|tuple[float,...] = 0.5, 
        std:float|tuple[float,...] = 0.5):
    """
    Create a list of transforms for image processing.
    Args:
        resize (int|tuple[int,int]): Resize image to this size. If None, no resize will be performed.
        crop_size (tuple[int,...]): Crop image to this size. If None, no cropping will be performed.
        crop_model (str): Crop model. Can be 'center' or 'random'. Defaults to 'center'. if None, no cropping will be performed.
        color_space (str): Color space of the image. Can be 'rgb', 'gray', 'hsv', etc. Defaults
        mean (float|tuple[float,...]): Mean value for normalization. Defaults to 0.5.
        std (float|tuple[float,...]): Std value for normalization. Defaults to 0.5.
    Returns:
        list: A list of transforms.
    """
    
    # Convert crop_size to tuple if it's int
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    
    transform_list = []
    # resize image
    if resize:
        transform_list.append(transforms.Resize(resize))
    # crop image
    if crop_model:
        crop_model = crop_model.lower()
    if crop_model == "random":
        transform_list.append(transforms.RandomCrop(crop_size))
    elif crop_model == "center":
        transform_list.append(transforms.CenterCrop(crop_size))
    else:
        pass # 不裁剪
    # Add color space conversion if needed
    cs = color_space.lower()
    if cs in ("rgb",):
        pass  # 默认RGB，无需转换
    elif cs in ("gray", "grey", "grayscale", "l"):
        transform_list.append(transforms.Grayscale())
    elif cs in ("hsv",):
        # torchvision 没有直接的 hsv 转换，需自定义或使用 lambda
        def _convert_hsv(img:Image.Image):
            return img.convert("HSV")
        transform_list.append(_convert_hsv)
    else:
        raise ValueError(f"unsupported color_space: {color_space}")
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform_list

def create_image_folder(
        root:str, 
        crop_size:int|tuple[int,int]=224,
        crop_model:str = "center",
        color_space:str = "rgb",
        mean:tuple[float,...] = (0.485, 0.456, 0.406),
        std:tuple[float,...] = (0.229, 0.224, 0.225),
        resize:int|tuple[int,int]=None,):
    
    # 如果 root 是 URL，则自动下载
    if root.startswith(("http://","https://")):
        dl_dir = auto_cache_download_list(root)
        root = dl_dir

    # Build transform pipeline based on crop_model
    transform_list = create_image_transforms(resize=resize, crop_size=crop_size, crop_model=crop_model, color_space=color_space, mean=mean, std=std)
    return datasets.ImageFolder(root, transforms.Compose(transform_list))


class ImageFolder(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dataset = create_image_folder(*args, **kwargs)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

class LHImageDataset(Dataset):
    def __init__(
            self,
            hd_folder:str,
            low_folder:str,
            color_space:str = "rgb",
            mean:tuple[float,...] = (0.5,0.5,0.5),
            std:tuple[float,...] = (0.5,0.5,0.5),
        ):
        super().__init__()
        # 查找HD_FOLDER 下的图片
        
        self.hd_folder = hd_folder
        self.low_folder = low_folder
        # 查找Low_FOLDER 下的图片
        hd_files:list[str] = os.listdir(hd_folder)
        low_files:list[str] = os.listdir(low_folder)
        common_files:list[str] = [f for f in hd_files if f in low_files]
        self.filenames = common_files
        # 构建转换流水线
        transform_list = create_image_transforms(
            resize=None, 
            color_space=color_space, 
            crop_model=None, # 不裁剪
            mean=mean, 
            std=std)
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        hd_img = Image.open(os.path.join(self.hd_folder, self.filenames[index])).convert("RGB")
        low_img = Image.open(os.path.join(self.low_folder, self.filenames[index])).convert("RGB")
        hd_data = self.transform(hd_img)
        low_data = self.transform(low_img)
        return low_data, hd_data


if __name__ == "__main__":
    dataset = LHImageDataset(
        "D:/work/datasets/Image Super Resolution/dataset/train/high_res", 
        "D:/work/datasets/Image Super Resolution/dataset/train/low_res")
    
    print(len(dataset))
    data = dataset[0]
    print(data[0].shape)
    print(data[1].shape)