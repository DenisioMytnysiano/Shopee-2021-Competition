import torch
import optuna
import pandas as pd
from typing import NoReturn
from ..common import get_neighbours
from ..common import getMetric


def get_best_params_bayesian(valid_embeddings: torch.Tensor,
                             validation_data: pd.DataFrame) -> NoReturn:

    def objective(trial: optuna.Trial) -> float:
        neighbours_count = trial.suggest_int("neighbours_counts", 40, 60)
        cosine_similarity = trial.suggest_float("cosine_similarity", 0.2, 0.5)
        valid_predictions = get_neighbours(
                    validation_data,
                    valid_embeddings.detach().cpu().numpy(),
                    threshold=cosine_similarity,
                    knn=neighbours_count
                )
        validation_data['oof'] = valid_predictions
        validation_data['f1'] = validation_data.apply(getMetric('oof'), axis=1)
        valid_f1 = validation_data.f1.mean()
        return valid_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
