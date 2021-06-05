import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import tqdm.notebook as tqdm
from ..common import AverageMeter
device = torch.device('cuda')


def train_epoch(train_dataset: DataLoader, model: nn.Module,
                criterion: nn.Module, optimizer: Optimizer, epoch: int,
                lr_scheduler: _LRScheduler = None):

    model.train()
    loss = AverageMeter()

    progress_bar_iter = tqdm(
        enumerate(train_dataset),
        total=len(train_dataset)
    )
    progress_bar_iter.set_description("Training")
    for batch_index, batch_data in progress_bar_iter:

        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        targets = batch_data["target"].to(device)

        output = model(input_ids, attention_mask)
        batch_loss = criterion(output, targets)
        batch_loss.backward()
        optimizer.first_step(zero_grad=True)
        loss.update(batch_loss.item(), len(batch_data))

        output = model(input_ids, attention_mask)
        batch_loss = criterion(output, targets)
        batch_loss.backward()
        optimizer.second_step(zero_grad=True)

        optimizer.zero_grad()

        if lr_scheduler is not None:
            progress_bar_iter.set_postfix(
                epoch=epoch,
                lr=optimizer.param_groups[0]['lr'],
                loss=loss.avg
            )
            lr_scheduler.step()
        else:
            progress_bar_iter.set_postfix(
                epoch=epoch,
                loss=loss.avg
            )
    return loss.avg
