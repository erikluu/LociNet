import torch
import torch.nn.functional as F
from tqdm import tqdm


def similarity_scores(matrix0: torch.Tensor, matrix1: torch.Tensor = None):
    """
    Compute cosine similarity between two sets of embeddings.

    Args:
        matrix0 (torch.Tensor): Tensor containing the first set of embeddings.
        matrix1 (torch.Tensor, optional): Tensor containing the second set of embeddings. If None, computes similarity within matrix0.
    
    Returns:
        torch.Tensor: A tensor where each row corresponds to an embedding in matrix0,
                      and contains the cosine similarity metric with matrix1 if provided.
    """
    if not matrix1:
        similarity_matrix = F.cosine_similarity(matrix0.unsqueeze(1), matrix0.unsqueeze(0), dim=2)
        return similarity_matrix
    
    assert matrix0.size(-1) == matrix1.size(-1), f"Dimensions of embeddings1 ({matrix0.size(-1)}) and matrix1 ({matrix1.size(-1)}) do not match."
    similarity_matrix = F.cosine_similarity(matrix0.unsqueeze(1), matrix1.unsqueeze(0), dim=2)
    return similarity_matrix


def batch_similarity_scores(matrix, batch_size=32):
    similarity_matrix = torch.empty(0)
    for i in tqdm(range(0, len(matrix), batch_size), desc="Embedding Similarity"):
        batch_similarity_matrix = torch.empty(0)
        batch0 = matrix[i:i+batch_size]
        for j in range(0, len(matrix), batch_size):
            batch1 = matrix[j:j+batch_size]
            _, batch_batch_similarity_matrix = similarity_scores(batch0, batch1) # haha
            batch_similarity_matrix = torch.cat((batch_similarity_matrix, batch_batch_similarity_matrix), dim=1)
    
        similarity_matrix = torch.cat((similarity_matrix, batch_similarity_matrix))

    return similarity_matrix


if __name__ == "__main__":
    matrix0 = torch.randn(100, 1000)
    matrix1 = torch.randn(100, 1000)
    _, sim_mat = similarity_scores(matrix0)
    batch_sim_mat = batch_similarity_rankings(matrix0)
    assert torch.equal(sim_mat, batch_sim_mat)