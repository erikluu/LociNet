import torch
import torch.nn.functional as F

def similarity_rankings_2d(em, k=None):
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


def batch_similarity_rankings_generator(em, batch_size=32):
    for i in range(0, len(em), batch_size):
        batch = em[i:i+batch_size]
        _, batch_similarity_matrix = similarity_rankings_2d(batch)
        yield batch_similarity_matrix


def batch_similarity_rankings(em, batch_size=32, save_path=None):
    similarity_matrices = []
    for similarity_matrix in batch_similarity_rankings(em, batch_size=32):
        similarity_matrices.append(similarity_matrix)
    
    return torch.cat(similarity_matrices, dim=0)


def argsort(matrix, k=None):
    rankings = torch.argsort(matrix, dim=1, descending=True)
    return rankings[:, :k] if k else rankings


if __name__ == "__main__":
    matrix1 = torch.randn(384, 10000)
    matrix2 = torch.randn(384, 10000)
    _, sim_mat = similarity_rankings_2d(matrix1, matrix2)
    batch_sim_mat = batch_similarity_rankings(matrix1, matrix2, batch_size=32)

    assert torch.equal(sim_mat, batch_sim_mat)