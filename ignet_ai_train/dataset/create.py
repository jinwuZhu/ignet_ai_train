from typing import Any
import torchvision.datasets as datasets
import ignet_ai_train.dataset.modules as ignet_datasets

_registered_datasets:dict[str,Any] = dict()

def _init_registered_datasets():
    global _registered_datasets
    from_packages = [datasets,ignet_datasets]
    for from_ in from_packages:
        for name in dir(from_):
            if name.startswith("_"):
                continue
            obj = getattr(from_, name)
            if isinstance(obj, type):
                _registered_datasets[name] = obj
    #  other datasets
    from ignet_ai_train.dataset.modules.audio_folder import get_dataset as audio_mels_folder
    # 加载音频文件，转换为梅尔谱图 Shape(N, 1, H, W)
    _registered_datasets["audio_mels_folder"] = audio_mels_folder


def create_dataset(name,*args,**kwargs):
    if not _registered_datasets: _init_registered_datasets()
    if name in _registered_datasets:
        return _registered_datasets[name](*args,**kwargs)
    else:
        raise Exception(f"Unknown dataset: {name}")

if __name__ == "__main__":
    dataset = create_dataset(
        "image_folder",
        root = "D:/work/datasets/dogcat/test_set/test_set",
        crop_size=256,
        color_space="rgb",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
    print(len(dataset))
    print(dataset[0][0].shape) # torch.Size([3, 224, 224])

