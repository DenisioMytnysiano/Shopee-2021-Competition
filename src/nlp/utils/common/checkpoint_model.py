import torch
from torch.optim import Optimizer
from torch.nn import Module
from typing import NoReturn, Callable, Union


def checkpoint_model(
    model: Module,
    loss: Union[Callable, Module],
    epoch: int,
    optimizer: Optimizer,
    path_to_checkpoint: str,
) -> NoReturn:
    """Method for creating a checkpoint for a model

    Args:
        model (Module): pytorch model
        loss (Union[Callable, Module]): loss function
        epoch (int): epoch number (used for training continuation)
        optimizer (Optimizer): your network optimizer
        path_to_checkpoint (str): path where yoy want to store model checkpoint

    Returns:
        NoReturn
    """

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path_to_checkpoint,
    )
