import requests
import os

def load_anypath_resources_text(url_or_path: str):
    """
    加载资源路径为文本
    Args:
        path: 资源路径, URL or Path
    Returns:
        text: 文本
    """
    if url_or_path.startswith(("http://", "https://")):
        r = requests.get(url_or_path)
        r.raise_for_status()
        text = r.text
    elif os.path.exists(url_or_path):
        text = open(url_or_path, "r", encoding='utf8').read()
    return text

import torch
from ignet_ai_train.model_builder import build_model

def load_model_pt(path:str):
    ckpt = torch.load(path)
    model_structure_config = ckpt['model_structure_config']
    model_args = ckpt.get('model_args', {}) or {}
    model = build_model(model_structure_config, **model_args)
    model.load_state_dict(ckpt['model'])
    return model