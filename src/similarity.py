import torch
import torch.nn.functional as F
from tqdm import tqdm

def soft_cosine_similarity(batch0, batch1):
    norm_batch0 = F.normalize(batch0, p=2, dim=1)
    norm_batch1 = F.normalize(batch1, p=2, dim=1)
    return norm_batch0 @ norm_batch1.t()


def euclidean_similarity(batch0, batch1, sigma=1.0):
    distances = torch.cdist(batch0, batch1, p=2)
    similarities = torch.exp(-distances ** 2 / (2 * sigma ** 2))
    return similarities


def similarity_scores(batch0, batch1, metric, sigma=1.0):
    if metric == "cosine":
        return F.cosine_similarity(batch0.unsqueeze(1), batch1.unsqueeze(0), dim=2)
    elif metric == "euclidean":
        return euclidean_similarity(batch0, batch1, sigma)
    elif metric == "soft_cosine":
        return soft_cosine_similarity(batch0, batch1)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def batch_similarity_scores(matrix, metric="cosine", batch_size=256):
    n = matrix.size(0)
    similarity_matrix = None

    for i in tqdm(range(0, n, batch_size), desc=f"Calculating {metric} similarities"):
        batch0 = matrix[i:i + batch_size]
        batch_similarity_matrix = None

        for j in range(0, n, batch_size):
            batch1 = matrix[j:j + batch_size]

            if batch_similarity_matrix is None:
                batch_similarity_matrix = similarity_scores(batch0, batch1, metric)
            else:
                batch_batch_similarity_matrix = similarity_scores(batch0, batch1, metric)
                batch_similarity_matrix = torch.cat((batch_similarity_matrix, batch_batch_similarity_matrix), dim=1)

        if similarity_matrix is None:
            similarity_matrix = batch_similarity_matrix
        else:
            if similarity_matrix.size(1) != batch_similarity_matrix.size(1): # pyright: ignore
                difference = similarity_matrix.size(1) - batch_similarity_matrix.size(1) # pyright: ignore
                padding = (0, abs(difference)) if difference < 0 else (0, 0)
                batch_similarity_matrix = F.pad(batch_similarity_matrix, padding) # pyright: ignore
                similarity_matrix = F.pad(similarity_matrix, (-difference, 0))
            similarity_matrix = torch.cat((similarity_matrix, batch_similarity_matrix)) # pyright: ignore

    return similarity_matrix


def get_all_similarities(embeddings):
    cosine_similarity = batch_similarity_scores(embeddings, metric="cosine")
    soft_cosine_similarity = batch_similarity_scores(embeddings, metric="soft_cosine")
    euclidean_similarity = batch_similarity_scores(embeddings, metric="euclidean")

    return cosine_similarity, soft_cosine_similarity, euclidean_similarity

# if __name__ == "__main__":
#     matrix0 = torch.randn(100, 1000)
#     matrix1 = torch.randn(100, 1000)
#     sim_mat = similarity_scores(matrix0)
#     batch_sim_mat = batch_similarity_scores(matrix0)
#     assert torch.equal(sim_mat, batch_sim_mat) # pyright: ignore

#     a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
#     print(a)
#     print(similarity_scores(a, metric="euclidean"))
#     print(similarity_scores(a, metric="cosine")) 
