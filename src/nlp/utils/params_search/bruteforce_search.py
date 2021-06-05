import torch
import numpy as np
import tqdm.notebook as tqdm
import pandas as pd
from typing import Dict
from ..common import get_neighbours
from ..common import getMetric


def get_best_params_bruteforce(valid_embeddings: torch.Tensor,
                               neighbours_counts: np.array,
                               cosine_sim_thresholds: np.array,
                               validation_data: pd.DataFrame) -> Dict:
    best_f1 = 0
    best_params = {
        "similarity_threshold": None,
        "neighbours_count": None
    }
    progress_bar = tqdm(
        total=len(neighbours_counts)*len(cosine_sim_thresholds)
    )
    progress_bar.set_description("Searching for best thresholds (BF)")
    for neighbours in neighbours_counts:
        for threshold in cosine_sim_thresholds:
            valid_predictions = get_neighbours(
                validation_data,
                valid_embeddings.detach().cpu().numpy(),
                threshold=threshold,
                knn=neighbours
            )
            validation_data['oof'] = valid_predictions
            validation_data['f1'] = validation_data.apply(
                getMetric('oof'),
                axis=1
            )
            valid_f1 = validation_data.f1.mean()
            if valid_f1 > best_f1:
                best_f1 = valid_f1
                best_params["similarity_threshold"] = threshold
                best_params["neighbours_count"] = neighbours
            progress_bar.set_postfix(
                best_f1=best_f1,
                threshold=threshold,
                neighbours=neighbours
            )
            progress_bar.update()
    print(f"Best F1 score:{best_f1}")
    print(best_params)
