
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import numpy as np
from typing import Callable, Tuple, Dict, Any
import tqdm

from skimage.metrics import (
    peak_signal_noise_ratio as psnr_metric,
    structural_similarity as ssim_metric
)


@torch.no_grad()
def evaluate_loss(
    model:nn.Module, 
    dataset:Dataset|DataLoader, 
    criterion,
    device:str = "cpu",
    verbose:bool = True,
    **kwargs
)->float|torch.Tensor:
    if isinstance(dataset, DataLoader):
        dataloader = dataset
    else:
        batch_size = kwargs.get("batch_size", 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device(device)
    # 验证集效果
    total_eval_num = 0
    val_loss = 0.0
    avg_loss = 0.0
    def _desc_message()->str: 
        return f"Eval {avg_loss:.3f}"
    dataloader:tqdm.tqdm[tuple[torch.Tensor,torch.Tensor|Any]] = tqdm.tqdm(dataloader, disable=not verbose, desc=_desc_message())
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss:torch.Tensor = criterion(pred, y)
        val_loss += loss.item()
        total_eval_num += 1
        avg_loss = val_loss / total_eval_num
        dataloader.set_description(_desc_message())
    return avg_loss



@torch.no_grad()
def evaluate_classification(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    topk: Tuple[int, ...] = (1,),
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device = None,
    postprocess: Callable = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    评估分类模型：计算 Top-k 准确率 + 混淆矩阵 + 各类别详细指标。

    Args:
        model: 已训练的 PyTorch 分类模型。
        dataset: 数据集，__getitem__ 返回 (input_tensor, label_int)。
        num_classes: 类别总数（必须提供）。
        topk: 要评估的 Top-k 值，如 (1, 5)。
        batch_size: DataLoader 批大小。
        num_workers: DataLoader 子进程数。
        device: 运行设备（如 'cuda', 'cpu'）。若为 None，自动选择。
        postprocess: 可选的，对预测结果进行后处理。输入模型的预测输出，返回处理后的预测logits。
        verbose: 是否打印进度条。

    Returns:
        dict 包含:
            - 'top1', 'top5', ... : float
            - 'confusion_matrix': np.ndarray of shape [num_classes, num_classes]
            - 'total_samples': int
            - 'details': dict, 键为类别索引（int），值为包含以下指标的字典：
                - 'auc': float 或 None（若无法计算）   # 各类别 One-vs-Rest ROC AUC
                - 'support': int                     # 该类别真实样本数量
                - 'precision': float                 # 精确率 / 查准率
                - 'recall': float                    # 召回率
                - 'f1_score': float                  # F1 分数
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )

    def _get_desc_message(topk, correct, total):
        if total == 0 or correct is None:
            tops = [f'Top-{k}: N/A' for k in topk]
        else:
            tops = [f'Top-{k}: {correct[k] / total:.3f}' for k in topk]
        return ' '.join(tops)

    dataloader:tqdm.tqdm[tuple[torch.Tensor,torch.Tensor|Any]] = tqdm.tqdm(dataloader, disable=not verbose, desc='Eval ' + _get_desc_message(topk, None, None), total=len(dataloader))
    maxk = max(topk)
    total = 0
    correct = {k: 0 for k in topk}

    all_pred_top1 = []
    all_labels = []
    all_probs = []   # 收集所有样本的预测概率

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels: torch.Tensor = labels.to(device)
        outputs: torch.Tensor | Any = model(inputs)  # [B, C]
        if postprocess is not None:
            outputs: torch.Tensor = postprocess(outputs)

        # 计算预测概率（softmax）
        probs = torch.softmax(outputs, dim=1)       # [B, C]

        # --- Top-k 准确率 ---
        _, pred_topk = outputs.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
        labels_view = labels.view(-1, 1).expand_as(pred_topk)

        for k in topk:
            hits = pred_topk[:, :k].eq(labels_view[:, :k]).any(dim=1)  # [B]
            correct[k] += hits.sum().item()

        total += labels.size(0)

        # --- 收集 Top-1 预测（用于混淆矩阵）及概率、标签 ---
        all_pred_top1.append(pred_topk[:, 0].cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

        desc_message = 'Eval ' + _get_desc_message(topk, correct, total)
        dataloader.set_description(desc_message)
    # 合并所有预测、标签和概率
    y_pred = torch.cat(all_pred_top1).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()      # shape: [total_samples, num_classes]

    # 构建混淆矩阵（确保包含所有类别，即使某些类未出现）
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes)
    )

    # 计算每类的 precision, recall, f1, support
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred,
        labels=np.arange(num_classes),
        average=None,
        zero_division=0
    )

    # 计算每类的 AUC (One-vs-Rest)
    aucs = []
    for i in range(num_classes):
        y_true_bin = (y_true == i).astype(int)
        y_score = y_prob[:, i]
        # 若某类只有一个类别（全正或全负），则无法计算 AUC
        if len(np.unique(y_true_bin)) < 2:
            aucs.append(None)
        else:
            aucs.append(roc_auc_score(y_true_bin, y_score))

    # 组合细节
    details = {}
    if verbose:
        print("Details:")
        print("CLASS\tAUC\tSUPPORT\tPRECISION\tRECALL\tF1_SCORE")
    for i in range(num_classes):
        details[i] = {
            'auc': aucs[i],
            'support': int(sup[i]),
            'precision': float(prec[i]),
            'recall': float(rec[i]),
            'f1_score': float(f1[i]),
        }
        if verbose:
            print(f"{i:3d}\t{aucs[i]:.3f}\t{sup[i]:d}\t{prec[i]:.3f}\t{rec[i]:.3f}\t{f1[i]:.3f}")
    # 计算准确率
    acc_dict = {f'top{k}': correct[k] / total for k in topk}
    results = {
        **acc_dict,
        'confusion_matrix': cm,
        'total_samples': total,
        'details': details
    }
    return results


def evaluate_super_resolution(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device = None,
    postprocess: Callable = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    
    # 1. 设备选择
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()

    # 2. 数据加载
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False
    )

    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    def _get_desc_message(psnr, ssim):
        return f"PSNR: {psnr:.3f}, SSIM: {ssim:.3f}"
    # 进度条
    pbar = tqdm.tqdm(loader, disable=not verbose, desc="Eval " + _get_desc_message(0, 0))

    with torch.no_grad():
        for lr_img, hr_img in pbar:
            lr_img:torch.Tensor = lr_img.to(device)
            hr_img:torch.Tensor = hr_img.to(device)

            # 模型前向传播
            sr_img:torch.Tensor = model(lr_img)

            # 如果有后处理（如去标准化、裁剪、数值截断等）
            if postprocess:
                sr_img = postprocess(sr_img)

            # 转换为 Numpy 以计算指标
            # 假设输入是 [B, C, H, W] 的 Tensor，值范围 [0, 1]
            sr_imgs_np = sr_img.cpu().numpy().transpose(0, 2, 3, 1)
            hr_imgs_np = hr_img.cpu().numpy().transpose(0, 2, 3, 1)

            # 遍历 Batch 计算指标
            for i in range(sr_imgs_np.shape[0]):
                # 确保数值限制在 [0, 1] 以防计算 PSNR 出错
                s_img = np.clip(sr_imgs_np[i], 0, 1)
                h_img = np.clip(hr_imgs_np[i], 0, 1)

                # 计算 PSNR (data_range 设置为 1.0)
                total_psnr += psnr_metric(h_img, s_img, data_range=1.0)
                
                # 计算 SSIM (对于彩色图像需要处理 channel 维度)
                # 如果是灰度图，multichannel=False；如果是彩色，使用 channel_axis=2
                is_multichannel = h_img.shape[-1] > 1
                total_ssim += ssim_metric(
                    h_img, s_img, 
                    data_range=1.0, 
                    channel_axis=2 if is_multichannel else None
                )
                
                total_samples += 1
            
            # 更新进度条
            psnr = total_psnr / total_samples if total_samples > 0 else 0
            ssim = total_ssim / total_samples if total_samples > 0 else 0
            desc_message = 'Eval ' + _get_desc_message(psnr, ssim)
            pbar.set_description(desc_message)

    # 3. 汇总结果
    results = {
        'psnr': total_psnr / total_samples if total_samples > 0 else 0,
        'ssim': total_ssim / total_samples if total_samples > 0 else 0,
        'total_samples': total_samples
    }

    if verbose:
        print(f"\nEvaluation Results: PSNR: {results['psnr']:.4f}, SSIM: {results['ssim']:.4f}")

    return results

