import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm.notebook as tqdm
from utils.common import AverageMeter
device = torch.device('cuda')


def validation_epoch(validation_dataset: DataLoader,
                     model: nn.Module, criterion: nn.Module, epoch: int):
    model.eval()
    loss = AverageMeter()
    with torch.no_grad():
        progress_bar_iter = tqdm(
            enumerate(validation_dataset),
            total=len(validation_dataset)
        )
        progress_bar_iter.set_description("Validation")

        for val_batch_index, val_batch_data in progress_bar_iter:

            val_input_ids = val_batch_data["input_ids"].to(device)
            val_attention_mask = val_batch_data["attention_mask"].to(device)
            targets = val_batch_data["target"].to(device)

            output = model(val_input_ids, val_attention_mask)
            val_loss = criterion(output, targets)

            loss.update(val_loss.item(), len(val_batch_data))
            progress_bar_iter.set_postfix(
                epoch=epoch,
                loss=loss.avg
            )
    return loss.avg
