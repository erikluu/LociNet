import torch
import torch.nn.functional as F
from tqdm import tqdm


def argsort(matrix: torch.Tensor, k: int = None) -> torch.Tensor:
    rankings = torch.argsort(matrix, dim=1, descending=True)
    return rankings[:, :k] if k else rankings


def similarity_rankings(em: torch.Tensor, k=None):
    """
    Compute top-k cosine similarity between a set of embeddings and itself.

    Args:
        em (torch.Tensor): Tensor containing the first set of embeddings.
        k (int, optional): Number of top similarities to retrieve for each embedding in em0. If None, all similarities are returned.
    Returns:
        torch.Tensor: A tensor where each row corresponds to an embedding in em,
                      and contains the indices of the top-k most similar embeddings in em1.
        torch.Tensor: A tensor where each row corresponds to an embedding in em,
                      and contains the direct similarity metric in em1.
    """
    similarity_matrix = F.cosine_similarity(em.unsqueeze(1), em.unsqueeze(0), dim=2)
    rankings = torch.argsort(similarity_matrix, dim=1, descending=True) # doesn't sort similarity_matrix
    rankings = rankings[:, :k] if k else rankings
    
    return rankings, similarity_matrix


def similarity_rankings_helper(em0, em1, k=None):
    """
    Just takes two parameters instead
    """
    assert em0.size(-1) == em1.size(-1), f"Dimensions of em0 ({em0.size(-1)}) and em1 ({em1.size(-1)}) do not match."

    similarity_matrix = F.cosine_similarity(em0.unsqueeze(1), em1.unsqueeze(0), dim=2)
    rankings = torch.argsort(similarity_matrix, dim=1, descending=True) # doesn't sort similarity_matrix
    rankings = rankings[:, :k] if k else rankings
    
    return rankings, similarity_matrix


def batch_similarity_rankings(em, batch_size=32):
    similarity_matrix = torch.empty(0)
    for i in tqdm(range(0, len(em), batch_size), desc="Embedding Similarity"):
        batch_similarity_matrix = torch.empty(0)
        batch0 = em[i:i+batch_size]
        for j in range(0, len(em), batch_size):
            batch1 = em[j:j+batch_size]
            _, batch_batch_similarity_matrix = similarity_rankings_helper(batch0, batch1) # haha
            batch_similarity_matrix = torch.cat((batch_similarity_matrix, batch_batch_similarity_matrix), dim=1)
    
        similarity_matrix = torch.cat((similarity_matrix, batch_similarity_matrix))

    return similarity_matrix


if __name__ == "__main__":
    matrix1 = torch.randn(100, 1000)
    matrix2 = torch.randn(100, 1000)
    _, sim_mat = similarity_rankings(matrix1)
    batch_sim_mat = batch_similarity_rankings(matrix1)
    assert torch.equal(sim_mat, batch_sim_mat)