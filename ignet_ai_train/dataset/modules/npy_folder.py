import numpy as np
import torch
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, Optional, Tuple

from ignet_ai_train.utils.download import auto_cache_download_list

def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x)

class NpyFolder(DatasetFolder):
    """
    Npy 加载类，直接继承 DatasetFolder。
    """
    def __init__(
        self,
        root: str,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        transform: Optional[Callable] = _to_tensor,
        loader: Callable[[str], Any] = np.load,
    ):
        # 如果没有指定验证函数，默认检查 .npy 扩展名
        extensions = ('.npy',) if is_valid_file is None else None
        
        if root.startswith(("http://","https://")):
            dl_dir = auto_cache_download_list(root)
            root = dl_dir

        super().__init__(
            root, 
            loader, 
            extensions=extensions,
            transform=transform,
            is_valid_file=is_valid_file
        )

if __name__ == '__main__':
    # 使用示例
    try:
        dataset = NpyFolder("D:/work/ignet_ai/cache/crash/linux")
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            x, label = dataset[0]
            print(f"Sample shape: {x.shape}, Label: {label}")
    except Exception as e:
        print(f"Error: {e}")