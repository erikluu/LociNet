import torch
import torch.nn.functional as F
from tqdm import tqdm


def similarity_scores(em: torch.Tensor):
    """
    Compute top-k cosine similarity between a set of embeddings and itself.

    Args:
        em (torch.Tensor): Tensor containing the first set of embeddings.
    Returns:
        torch.Tensor: A tensor where each row corresponds to an embedding in em,
                      and contains the direct similarity metric in em1.
    """
    similarity_matrix = F.cosine_similarity(em.unsqueeze(1), em.unsqueeze(0), dim=2)    
    return similarity_matrix


def similarity_rankings_helper(mat0, mat1, k=None):
    """
    Just takes two parameters instead
    """
    assert mat0.size(-1) == mat1.size(-1), f"Dimensions of em0 ({mat0.size(-1)}) and em1 ({mat1.size(-1)}) do not match."

    similarity_matrix = F.cosine_similarity(mat0.unsqueeze(1), mat1.unsqueeze(0), dim=2)
    rankings = torch.argsort(similarity_matrix, dim=1, descending=True) # doesn't sort similarity_matrix
    rankings = rankings[:, :k] if k else rankings
    
    return rankings, similarity_matrix


def batch_similarity_rankings(mat, batch_size=32):
    similarity_matrix = torch.matpty(0)
    for i in tqdm(range(0, len(mat), batch_size), desc="Embedding Similarity"):
        batch_similarity_matrix = torch.matpty(0)
        batch0 = mat[i:i+batch_size]
        for j in range(0, len(mat), batch_size):
            batch1 = mat[j:j+batch_size]
            _, batch_batch_similarity_matrix = similarity_rankings_helper(batch0, batch1) # haha
            batch_similarity_matrix = torch.cat((batch_similarity_matrix, batch_batch_similarity_matrix), dim=1)
    
        similarity_matrix = torch.cat((similarity_matrix, batch_similarity_matrix))

    return similarity_matrix


if __name__ == "__main__":
    matrix0 = torch.randn(100, 1000)
    matrix1 = torch.randn(100, 1000)
    _, sim_mat = similarity_scores(matrix0)
    batch_sim_mat = batch_similarity_rankings(matrix0)
    assert torch.equal(sim_mat, batch_sim_mat)