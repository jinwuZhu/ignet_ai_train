import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import requests

from ignet_ai_train.utils import train, TRAIN_EVENT_EPOCH_END, TRAIN_EVENT_STEP, TRAIN_EVENT_END
from ignet_ai_train.dataset.create import create_dataset
from ignet_ai_train.model_builder import build_model_by_yaml, create_model
from ignet_ai_train.optim.create import create_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model based on a YAML configuration file")
    parser.add_argument("--config", "-c", required=True, help="Path to the YAML configuration file")
    parser.add_argument(
        "--checkpoints", "-k", default="./checkpoints",
        help="Directory where checkpoints will be saved (default: ./checkpoints)"
    )
    parser.add_argument(
        "--device", "-d", default=None,
        help="Device to use, e.g. 'cpu' or 'cuda'. If omitted the script will choose automatically."
    )
    parser.add_argument(
        "--resume", "-r", default=None,
        help="Optional checkpoint file to resume training from."
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf8") as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def make_object(name: str, creator: callable, default_kwargs: dict | None, extra_kwargs: dict | None = None):
    """Helper that creates an object using configuration dicts."""
    if not name:
        return None
    args = {}
    if default_kwargs:
        args.update(default_kwargs)
    if extra_kwargs:
        args.update(extra_kwargs)
    return creator(name, **args)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ---------- model ----------
    model_cfg = cfg.get("model", {})
    model_yaml = model_cfg.get("yaml")
    if not model_yaml:
        raise ValueError("`model.yaml` must be specified under `model` in configuration")

    # allow relative path by interpreting w.r.t. config file
    config_dir = os.path.dirname(os.path.abspath(args.config))
    model_yaml_path = (
        model_yaml
        if os.path.isabs(model_yaml)
        else os.path.join(config_dir, model_yaml)
    )

    model_args = model_cfg.get("args", {})
    model = build_model_by_yaml(model_yaml_path, **model_args)

    # ---------- criterion ----------
    criterion = None
    if cfg.get("criterion"):
        crit_cfg = cfg["criterion"]
        crit_name = crit_cfg.get("name")
        crit_args = crit_cfg.get("args", {})
        if crit_name:
            criterion = create_model(crit_name, **crit_args)

    # ---------- optimizer ----------
    optimizer = None
    if cfg.get("optimizer"):
        opt_cfg = cfg["optimizer"]
        opt_name = opt_cfg.get("name")
        opt_args = opt_cfg.get("args", {})
        if opt_name:
            optimizer = create_optimizer(opt_name, model.parameters(), **opt_args)

    # ---------- datasets ----------
    if not cfg.get("train_dataset"):
        raise ValueError("`train_dataset` section is required in configuration")
    td_cfg = cfg["train_dataset"]
    train_dataset = create_dataset(td_cfg["name"], **td_cfg.get("args", {}))

    eval_dataset = None
    if cfg.get("eval_dataset"):
        ed_cfg = cfg["eval_dataset"]
        eval_dataset = create_dataset(ed_cfg["name"], **ed_cfg.get("args", {}))

    # helper for validation
    def evaluate(dataset, device, criterion, batch_size):
        model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, label in loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                if criterion is not None:
                    loss_val = criterion(output, label).item()
                else:
                    loss_val = 0.0
                total_loss += loss_val
                count += 1
        model.train()
        return (total_loss / count) if count > 0 else 0.0

    # ---------- hyperparameters ----------
    epochs = cfg.get("epochs", 100)
    batch_size = cfg.get("batch_size", 16)

    # ---------- checkpoints & logging ----------
    ckpt_dir = args.checkpoints
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=os.path.join(ckpt_dir, "logs"))

    losses: list[float] = []
    min_loss = float("inf")

    # ---------- callbacks configuration ----------
    callbacks_config = cfg.get("callbacks", {})
    callback_urls = {
        TRAIN_EVENT_EPOCH_END: callbacks_config.get("epoch_end", []),
        TRAIN_EVENT_STEP: callbacks_config.get("train_step", []),
        TRAIN_EVENT_END: callbacks_config.get("train_end", []),
    }

    def post_callbacks(event, epoch, step, loss):
        """Send callback requests to configured URLs for the given event."""
        urls = callback_urls.get(event, [])
        if not urls:
            return
        callback_data = {
            "event": str(event),
            "epoch": int(epoch),
            "step": int(step),
            "loss": float(loss),
        }
        for url in urls:
            try:
                requests.post(url, json=callback_data, timeout=5)
            except Exception:
                pass
                # print(f" [Warning] Error posting to {url}")

    def train_callback(event, epoch, step, loss, model_obj, optim_obj, crit_obj):
        nonlocal losses, min_loss
        print(f"\r[{event}]Epoch: {epoch} Iter: {step} Loss: {loss:.4f}", end="")
        def _save_checkpoint(name):
            """
            Save checkpoint to disk
            Args:
                name (str): Name of the checkpoint
            Returns:
                str: Path to the saved checkpoint
            """
            filepath = os.path.join(ckpt_dir, f"{name}.pt")
            torch.save(
                {
                    "model": model_obj.state_dict(),
                    "epoch": epoch,
                    "min_loss": min_loss,
                    "loss": mean_loss,
                    "model_yaml": model_yaml,
                },
                filepath,
            )
            return filepath

        # post callbacks
        post_callbacks(event, epoch, step, loss)
        if event == TRAIN_EVENT_EPOCH_END:
            print()
            losses.append(loss)
            mean_loss = sum(losses) / len(losses)
            losses.clear()
            print(f"Save last model,mean_loss: {mean_loss}")
            _save_checkpoint("last")
            logger.add_scalar("loss_epoch", loss, epoch)
            if mean_loss < min_loss:
                min_loss = mean_loss
                _save_checkpoint("best")

            # perform evaluation if dataset provided
            if eval_dataset is not None:
                val_loss = evaluate(eval_dataset, device, criterion, batch_size)
                print(f"  eval_loss: {val_loss:.4f}")
                logger.add_scalar("loss_eval", val_loss, epoch)
        elif event == TRAIN_EVENT_END:
            _save_checkpoint("last")

    # ---------- prepare environment ----------
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print("parameters: ", sum(p.numel() for p in model.parameters()))
    print(f"Using {device}")

    # resume if requested
    if args.resume:
        if os.path.isfile(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
                min_loss = ckpt.get("min_loss", float("inf"))
            else:
                model.load_state_dict(ckpt)
            print(f"Resumed weights from {args.resume}")
        else:
            print(f"Resume file {args.resume} not found, ignoring")

    # ---------- run training ----------
    train(
        model,
        train_dataset,
        epoch=epochs,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=batch_size,
        callback=train_callback,
    )

    logger.close()


if __name__ == "__main__":
    main()
