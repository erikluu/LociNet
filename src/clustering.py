import torch
from sklearn.cluster import DBSCAN


def dbscan(embeddings: torch.Tensor, eps: float = 0.5, min_samples: int = 5, metric: str = "cosine"):
    """
    Perform DBSCAN clustering on the given embeddings.

    Args:
        embeddings (torch.Tensor): Input tensor containing the embeddings to be clustered.
    Returns:
        list[int]: A list of cluster labels for each embedding.
    """
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(embeddings)
    return labels
