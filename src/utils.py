import torch
import psutil
import networkx as nx
from pyvis.network import Network
from sklearn.decomposition import PCA


def pca(matrix: torch.Tensor, n_components: int = 2) -> list[tuple]:
    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    return pca.transform(matrix)


def argsort(matrix: torch.Tensor, k: int = 0) -> torch.Tensor:
    rankings = torch.argsort(matrix, dim=1, descending=True)
    return rankings[:, :k] if k else rankings


def sort_matrix_values(matrix: torch.Tensor, k: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_indices = torch.argsort(matrix, dim=1, descending=True)
    sorted_values = torch.gather(matrix, 1, sorted_indices)
    return sorted_indices[:, :k] if k else sorted_indices, sorted_values[:, :k] if k else sorted_values


def batch_size_estimate(em_size, num_em_sets=1):
    mem_req_per_embedding_bytes = em_size.size(-1) * 4 * num_em_sets # req memory in bytes
    mem_req_per_embedding_gbytes = mem_req_per_embedding_bytes / (1024**3) # req memory in GB

    mem = psutil.virtual_memory()
    available_mem = mem.available / (1024**3) # available memory in GB
    max_embeddings_in_memory = available_mem / mem_req_per_embedding_gbytes

    batch_size = max_embeddings_in_memory // 100 # set to a quarter of max
    return int(batch_size)
