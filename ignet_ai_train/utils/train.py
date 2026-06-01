import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from typing import Any, Callable
from ignet_ai_train.utils.eval import evaluate_loss

TRAIN_EVENT_EPOCH_END = "epoch_end"
TRAIN_EVENT_STEP = "train_step"
TRAIN_EVENT_END = "train_end"

def _train_print_progress(
    even:str,
    epoch:int,
    iter_index:int,
    loss:float, 
    model:nn.Module, 
    optimizer:torch.optim.Optimizer, 
    criterion,
    **kwargs
):
    print(f"\rEpoch: ({(epoch + 1):3d}|{iter_index:4d}); Loss: {loss:.4f}", end="")

def train(
    model:nn.Module, 
    dataset:Dataset|str,
    optimizer:torch.optim.Optimizer = None, 
    criterion = None,
    device:str = "cpu",
    epoch:int = 100,
    batch_size:int = 32,
    callback:Callable = _train_print_progress,
    eval_dataset:Dataset|str = None,
    scheduler:torch.optim.Optimizer|Any = None,
    **kwargs
)->None:
    """
    训练
    Args:
        model (nn.Module): 模型
        dataset (Dataset|str): 训练数据集
        optimizer (torch.optim.Optimizer, optional): 优化器. Defaults to None.
        criterion (torch.nn.Module, optional): 损失函数. Defaults to None.
        device (str, optional): 设备. Defaults to "cpu".
        epoch (int, optional): 训练轮数. Defaults to 100.
        batch_size (int, optional): 批次大小. Defaults to 32.
        callback (Callable, optional): 训练回调函数. Defaults to _train_print_progress.
        eval_dataset (Dataset|str, optional): 验证数据集. Defaults to None.
        scheduler (torch.optim.Optimizer|Any, optional): 调度器. Defaults to None.
        **kwargs:
            verbose (bool, optional): 是否打印进度. Defaults to False.
    Returns:
        None
    """
    model.train()
    if isinstance(dataset, str):
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = ImageFolder(dataset,trans)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = optimizer or torch.optim.SGD(model.parameters())
    model.to(device)
    
    def _callback(
        event:str,
        epoch:int,
        iter_index:int,
        loss:float, 
        model:nn.Module, 
        optimizer:torch.optim.Optimizer, 
        criterion,
        **kwargs
    ):
        if callback:
            callback(event,epoch,iter_index,loss,model,optimizer,criterion,**kwargs)

    for e in range(epoch):
        for i, (data, label) in enumerate(train_loader): # ValueError: too many values to unpack (expected 2)
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            _callback(event=TRAIN_EVENT_STEP,epoch=e, iter_index=i, loss=float(loss.item()), model=model, optimizer=optimizer, criterion=criterion)
            optimizer.step()
        if eval_dataset and scheduler:
            verbose = kwargs.get("verbose", False)
            val_loss = evaluate_loss(model, eval_dataset, criterion, device,verbose=verbose)
            scheduler.step(val_loss)
            _callback(event=TRAIN_EVENT_EPOCH_END, epoch=e, iter_index=i, loss=float(loss.item()), model=model, optimizer=optimizer, criterion=criterion,eval_loss=val_loss)
        else:
            _callback(event=TRAIN_EVENT_EPOCH_END, epoch=e, iter_index=i, loss=float(loss.item()), model=model, optimizer=optimizer, criterion=criterion)
    _callback(event=TRAIN_EVENT_END, epoch=e, iter_index=i, loss=float(loss.item()), model=model, optimizer=optimizer, criterion=criterion)