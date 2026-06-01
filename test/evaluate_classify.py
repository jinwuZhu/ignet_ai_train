
import argparse
import json
import os
import torch
import matplotlib.pyplot as plt

from ignet_ai_train.model_builder import build_model
from ignet_ai_train.dataset.create import create_dataset
from ignet_ai_train.utils.eval import evaluate_classification


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pt', required=True, help='Path to checkpoint .pt file')
    p.add_argument('--device', default=None, help="Device, e.g. 'cpu' or 'cuda'")
    p.add_argument('--dataset-name', default="image_folder", help='Dataset class name (e.g. image_folder)')
    p.add_argument('--dataset-args', default='{}', help='JSON string of args for dataset creator')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--topk', default='[1,5]', help='JSON list for top-k, e.g. "[1,5]"')
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # load checkpoint
    ckpt_path = args.pt
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)

    # extract model structure and args
    if 'model_structure_config' in ckpt and 'model_args' in ckpt:
        model_structure_config = ckpt['model_structure_config']
        model_args = ckpt.get('model_args', {}) or {}
    else:
        raise RuntimeError('Checkpoint must contain `model_structure_config` and `model_args` to rebuild the model')

    # ensure num classes available
    num_classes = model_args.get('nc')
    if num_classes is None:
        raise RuntimeError('`nc` (number of classes) must be present in model_args of the checkpoint')

    # build model
    model = build_model(model_structure_config, **model_args)

    # load weights
    state_dict = ckpt.get('model', ckpt)
    try:
        model.load_state_dict(state_dict)
    except Exception:
        # try non-strict load
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)

    # construct dataset
    try:
        ds_args = json.loads(args.dataset_args)
    except Exception:
        raise RuntimeError('`--dataset-args` must be a valid JSON string')

    dataset = create_dataset(args.dataset_name, **ds_args)

    topk = tuple(json.loads(args.topk))

    result = evaluate_classification(
        model=model,
        dataset=dataset,
        num_classes=int(num_classes),
        topk=topk,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        verbose=True,
    )

    # print results
    for k, v in result.items():
        if k.startswith('top'):
            print(f'{k}: {v*100:.3f}%')

    cm = result['confusion_matrix']
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save to model directory
    model_dir = os.path.dirname(ckpt_path)
    output_path = os.path.join(model_dir, 'confusion_matrix.png')
    plt.savefig(output_path)
    # Export to ONNX
    x_smaple,_ = dataset[0]
    # torch.shape [C,H,W] to [B,C,H,W]
    x_smaple = x_smaple.unsqueeze_(0)
    torch.onnx.export(model, x_smaple.to(device), os.path.join(model_dir, 'model.onnx'),opset_version=18,external_data=False)
    print(f'Saved confusion matrix to {output_path}')


if __name__ == '__main__':
    main()