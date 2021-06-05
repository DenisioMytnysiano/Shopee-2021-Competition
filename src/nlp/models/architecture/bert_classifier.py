import torch
from . import ArcMarginProduct
from config import NUM_GROUPS, MODEL_NAME
from transformers import AutoModel


class ShopeeBERTModel(torch.nn.Module):
    def __init__(self, dropout: float = 0.0, fc_dim: int = 512):
        super(ShopeeBERTModel, self).__init__()

        self.transformer = AutoModel.from_pretrained(MODEL_NAME)
        self.metric_in_features = self.transformer.config.hidden_size

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(self.metric_in_features, fc_dim),
            torch.nn.BatchNorm1d(fc_dim)
        )

        self.final = ArcMarginProduct(fc_dim, NUM_GROUPS)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def extract_features(self, input_ids, attention_mask):
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.mean_pooling(output, attention_mask)
        output = self.feature_extractor(output)
        return output

    def forward(self, input_ids, attention_mask):
        features = self.extract_features(input_ids, attention_mask)
        output = self.final(features)
        return output
