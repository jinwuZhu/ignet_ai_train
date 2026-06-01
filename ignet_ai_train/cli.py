import argparse
import os
from typing import Any
import yaml
import torch
import requests
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from ignet_ai_train.utils import train, TRAIN_EVENT_EPOCH_END, TRAIN_EVENT_STEP, TRAIN_EVENT_END
from ignet_ai_train.dataset.create import create_dataset
from ignet_ai_train.model_builder import build_model, create_model
from ignet_ai_train.optim.create import create_optimizer
from ignet_ai_train.utils.loader import load_anypath_resources_text

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a model based on a YAML configuration file")
    parser.add_argument("--config", "-c", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--checkpoints", "-k", default="./checkpoints", help="Directory for checkpoints")
    parser.add_argument("--device", "-d", default=None, help="cpu or cuda")
    parser.add_argument("--resume", "-r", default=None, help="Resume training from checkpoint")
    return parser.parse_args(argv)

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf8") as f:
        return yaml.safe_load(f) or {}

def save_checkpoint(path, model, optimizer, epoch, min_loss, mean_loss, model_args, model_structure_config, model_yaml):
    """提取独立的保存逻辑"""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "model_args": model_args,
        "model_structure_config": model_structure_config,
        "epoch": epoch,
        "min_loss": min_loss,
        "loss": mean_loss,
        "model_yaml": model_yaml,
    }
    torch.save(state, path)
    print(f" -> Checkpoint saved: {path} (Loss: {mean_loss:.4f})")

def evaluate(model, dataset, device, criterion, batch_size):
    """严谨的模型评估逻辑"""
    model.eval() # 必须切换到 eval 模式
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss_val = criterion(output, label).item() if criterion else 0.0
            total_loss += loss_val
            count += 1
    model.train() # 切回训练模式
    return (total_loss / count) if count > 0 else 0.0

# 定义全局变量，初始为 None
_http_session:requests.Session = None

def get_session():
    """延迟初始化 Session，确保只在需要时创建一次"""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
    return _http_session

def main(argv=None):
    global _http_session
    args:dict[str,Any] = parse_args(argv)
    cfg = load_config(args.config)
    
    # 0. 提前确定设备
    device:str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # 1. 模型构建
    model_cfg = cfg.get("model", {})
    model_yaml:str = model_cfg.get("yaml")
    if not model_yaml:
        raise ValueError("`model.yaml` missing in config")

    config_dir = os.path.dirname(os.path.abspath(args.config))
    model_yaml_path = model_yaml if (model_yaml.startswith("http") or os.path.isabs(model_yaml)) \
                      else os.path.join(config_dir, model_yaml)

    model_yaml_data = load_anypath_resources_text(model_yaml_path)
    model_structure_config = yaml.safe_load(model_yaml_data) or {}
    
    model_args = model_cfg.get("args", {})
    model = build_model(model_structure_config, **model_args).to(device) # 直接上设备

    # 2. 损失函数与优化器
    criterion = None
    if cfg.get("criterion"):
        c_cfg = cfg["criterion"]
        criterion = create_model(c_cfg["name"], **c_cfg.get("args", {})).to(device)

    optimizer = None
    if cfg.get("optimizer"):
        o_cfg = cfg["optimizer"]
        optimizer = create_optimizer(o_cfg["name"], model.parameters(), **o_cfg.get("args", {}))

    # 3. 数据集
    if "train_dataset" not in cfg:
        raise ValueError("`train_dataset` required")
    
    train_ds = create_dataset(cfg["train_dataset"]["name"], **cfg["train_dataset"].get("args", {}))
    eval_ds = None
    if cfg.get("eval_dataset"):
        eval_ds = create_dataset(cfg["eval_dataset"]["name"], **cfg["eval_dataset"].get("args", {}))

    # 4. 训练状态与日志
    ckpt_dir = args.checkpoints
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=os.path.join(ckpt_dir, "logs"))
    
    epochs = cfg.get("epochs", 100)
    batch_size = cfg.get("batch_size", 16)
    min_loss = float("inf")
    
    # 加载 Resume (优化：加载优化器状态以保证学习率等衔接)
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        if optimizer and "optimizer" in ckpt and ckpt["optimizer"]:
            optimizer.load_state_dict(ckpt["optimizer"])
        min_loss = ckpt.get("min_loss", float("inf"))
        print(f"Resumed from {args.resume} (Previous Min Loss: {min_loss:.4f})")

    # 5. 回调逻辑
    cb_urls = cfg.get("callbacks", {})

    def train_callback(event, epoch, step, loss, model_obj, optim_obj, crit_obj,**kwargs):
        nonlocal min_loss
        
        # 实时打印
        print(f"\rEpoch: {epoch:3d} , {step:5d} Loss: {loss:.4f}", end="")
        current_val_loss:float = None
        
        if event == TRAIN_EVENT_EPOCH_END:
            print() # 换行
            logger.add_scalar("Loss/Train_Epoch", loss, epoch)
            
            # 验证逻辑
            target_loss = loss
            if eval_ds:
                current_val_loss = evaluate(model_obj, eval_ds, device, crit_obj, batch_size)
                logger.add_scalar("Loss/Eval", current_val_loss, epoch)
                target_loss = current_val_loss
                print(f" >> Eval Loss: {current_val_loss:.4f}")

            # 保存最后一次
            save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model_obj, optim_obj, 
                            epoch, min_loss, target_loss, model_args, model_structure_config, model_yaml)

            # 保存最优解
            if target_loss < min_loss:
                min_loss = target_loss
                save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model_obj, optim_obj, 
                                epoch, min_loss, target_loss, model_args, model_structure_config, model_yaml)

        elif event == TRAIN_EVENT_END:
            save_checkpoint(os.path.join(ckpt_dir, "final.pt"), model_obj, optim_obj, 
                            epoch, min_loss, loss, model_args, model_structure_config, model_yaml)

        # HTTP Webhook 回调
        event_key = "epoch_end" if event == TRAIN_EVENT_EPOCH_END else \
                    "train_step" if event == TRAIN_EVENT_STEP else "train_end"
        
        urls = cb_urls.get(event_key, [])
        if urls:
            payload = {
                "event": str(event), "epoch": epoch, "step": step, 
                "loss": float(loss), "eval_loss": current_val_loss
            }
            for url in urls:
                try:
                    http_session = get_session()
                    http_session.post(url, json=payload, timeout=2) # 缩短超时，避免阻塞训练
                except Exception:
                    pass

    # 6. 开始训练
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    try:
        train(
            model, train_ds, epoch=epochs, device=device,
            criterion=criterion, optimizer=optimizer,
            batch_size=batch_size, callback=train_callback,
        )
    finally:
        logger.close()
        if _http_session:
            _http_session.close()
            _http_session = None

if __name__ == "__main__":
    main()