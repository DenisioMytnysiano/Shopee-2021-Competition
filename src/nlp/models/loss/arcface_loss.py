import math
import torch
import torch.nn as nn


class ArcFaceLoss(nn.Module):
    def __init__(self, m: float = 0.5,
                 s: float = 40.0,
                 ls_eps: float = 0.0,
                 weight: torch.Tensor = None,
                 class_weight_norm: str = 'global',
                 reduction: str = 'mean'):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.reduction = reduction
        self.weight = weight
        self.class_weight_norm = class_weight_norm
        self.ls_eps = ls_eps
        self.s = s

        self.sin_m = math.sin(m)
        self.cos_m = math.cos(m)
        self.cos_th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        cosine = logits.float()
        sine = torch.sqrt(1 - torch.pow(cosine, 2))

        phi = self.cos_m * cosine - self.sin_m * sine
        phi = torch.where(cosine > self.cos_th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + \
                        self.ls_eps / cosine.size(1)

        output = one_hot * phi + (1 - one_hot) * cosine
        output *= self.s

        loss = self.criterion(output, labels)
        if self.weight is not None:

            weights = self.weight[labels].to(logits.device)
            loss *= weights

            if self.class_weight_norm == "batch":
                return loss.sum() / weights.sum()
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
