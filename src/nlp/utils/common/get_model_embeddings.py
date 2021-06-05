import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm.notebook as tqdm
device = torch.device("cuda")


def get_bert_embeddings(validation_dataset: DataLoader,
                        model: nn.Module,
                        embeddings_size: int = 512):
    model.eval()
    bert_embeddings = torch.zeros(0, embeddings_size).to(device)
    progress_bar_iter = tqdm(
        enumerate(validation_dataset),
        total=len(validation_dataset)
    )
    progress_bar_iter.set_description("Extracting text embeddings")

    for val_batch_index, val_batch_data in progress_bar_iter:

        val_input_ids = val_batch_data["input_ids"].to(device)
        val_attention_mask = val_batch_data["attention_mask"].to(device)

        with torch.no_grad():
            embeddings = model.extract_features(
                val_input_ids,
                val_attention_mask
            )
        bert_embeddings = torch.cat((bert_embeddings, embeddings))

    gc.collect()
    torch.cuda.empty_cache()
    return bert_embeddings
