import gc
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_neighbors(df, embeddings: torch.Tensor,
                  knn: int = 50, threshold: float = 0.0):
    model = NearestNeighbors(n_neighbors=knn, metric='cosine')
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    preds = []
    for k in range(embeddings.shape[0]):
        idx = np.where(distances[k, ] < threshold)[0]
        ids = indices[k, idx]
        posting_ids = df['posting_id'].iloc[ids].values
        preds.append(posting_ids)

    gc.collect()
    torch.cuda.empty_cache()
    return preds
