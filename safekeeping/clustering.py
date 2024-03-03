import torch
from utils import argsort

# https://github.com/rusty1s/pytorch_cluster
# import torch_cluster as tc


def knn(matrix: torch.Tensor, k=10):
    rankings = argsort(matrix, k)
    nearest_neighbors = rankings[:, 1:k] # not including itself, or should there be a link to itself...
    return nearest_neighbors


# cooccurence of labels vs embeddings
# hierarchical clustering based on coocurrence of labels

