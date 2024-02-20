import torch
import psutil


def argsort(matrix: torch.Tensor, k: int = None) -> torch.Tensor:
    rankings = torch.argsort(matrix, dim=1, descending=True)
    return rankings[:, :k] if k else rankings


def batch_size_estimate(em_size, num_em_sets=1):
    mem_req_per_embedding_bytes = em_size.size(-1) * 4 * num_em_sets # req memory in bytes
    mem_req_per_embedding_gbytes = mem_req_per_embedding_bytes / (1024**3) # req memory in GB
    
    mem = psutil.virtual_memory()
    available_mem = mem.available / (1024**3) # available memory in GB
    max_embeddings_in_memory = available_mem / mem_req_per_embedding_gbytes

    batch_size = max_embeddings_in_memory // 100 # set to a quarter of max
    return int(batch_size)