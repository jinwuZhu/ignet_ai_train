import torch

from ignet_ai_train.model_builder import build_model

if __name__ == "__main__":
    ckpt = torch.load("checkpoints/best.pt")
    model_structure_config = ckpt['model_structure_config']
    model_args = ckpt.get('model_args', {}) or {}
    model = build_model(model_structure_config, **model_args)
    model.load_state_dict(ckpt['model'])
    torch.onnx.export(model, torch.randn(1, 3, 114, 114), "checkpoints/model.onnx", external_data=False)