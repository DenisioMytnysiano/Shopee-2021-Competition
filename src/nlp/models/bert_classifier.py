import torch
from models import ArcMarginProduct
from config import NUM_GROUPS, MODEL_NAME
from transformers import AutoModel
from typing import NoReturn


class ShopeeBERTModel(torch.nn.Module):
    def __init__(self, dropout: float = 0.2, fc_dim: int = 512) -> NoReturn:
        """
        Args:
            dropout (float, optional):
            Dropout probability in dropout layers. Defaults to 0.2.
            fc_dim (int, optional):
            Dimension of fully-connected layer of network. Defaults to 512.
        """
        super(ShopeeBERTModel, self).__init__()

        self.transformer = AutoModel.from_pretrained(MODEL_NAME)
        self.metric_in_features = self.transformer.config.hidden_size
        self.dropout = torch.nn.Dropout(p=dropout)
        self.fc1 = torch.nn.Linear(self.metric_in_features, fc_dim)
        self.fc2 = torch.nn.Linear(fc_dim, fc_dim * 10)
        self.bn1 = torch.nn.BatchNorm1d(fc_dim)
        self.bn2 = torch.nn.BatchNorm1d(fc_dim * 10)
        self.relu = torch.nn.ReLU()
        self._init_params()
        self.metric_in_features = fc_dim * 10
        self.final = ArcMarginProduct(self.metric_in_features, NUM_GROUPS)
        for child in self.transformer.children():
            for param in child.parameters():
                param.requires_grad = False

    def _init_params(self) -> NoReturn:
        """
        Layers weights initialization
        """
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.constant_(self.bn1.weight, 1)
        torch.nn.init.constant_(self.bn1.bias, 0)
        torch.nn.init.constant_(self.bn2.weight, 1)
        torch.nn.init.constant_(self.bn2.bias, 0)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, label: int
    ) -> torch.Tensor:
        """Forward pass of the network

        Args:
            input_ids (torch.Tensor): tokenized sentence
            attention_mask (torch.Tensor): attention mask of a sentence
            label (int): class label

        Returns:
            torch.Tensor: probability distribution over N classes
        """

        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = output[0]
        output = output[:, 0, :]
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.final(output, label)
        return output
