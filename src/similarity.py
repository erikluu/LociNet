import torch
import torch.nn.functional as F
from tqdm import tqdm

def compute_feature_similarity(features):
    features = features.t()  # Transpose the feature matrix to get features as rows
    return F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)


def soft_cosine_similarity(batch0, batch1, feature_similarity):
    assert feature_similarity.size(0) == batch0.size(1) and feature_similarity.size(1) == batch1.size(1), \
        "Feature similarity matrix dimensions do not match the feature dimensions of the batches"
    
    norm_batch0 = F.normalize(batch0, p=2, dim=1)
    norm_batch1 = F.normalize(batch1, p=2, dim=1)
    
    # Compute the soft cosine similarity
    soft_cos_sim = torch.zeros(batch0.size(0), batch1.size(0), device=batch0.device)
    for i in range(batch0.size(0)):
        for j in range(batch1.size(0)):
            # Compute the weighted inner product
            numerator = torch.matmul(torch.matmul(norm_batch0[i], feature_similarity), norm_batch1[j])
            denominator = torch.sqrt(torch.matmul(torch.matmul(norm_batch0[i], feature_similarity), norm_batch0[i])) * torch.sqrt(torch.matmul(torch.matmul(norm_batch1[j], feature_similarity), norm_batch1[j]))
            soft_cos_sim[i, j] = numerator / denominator
    
    return soft_cos_sim


def similarity_scores(batch0, batch1, metric, feature_similarity=None, sigma=1.0):
    if metric == "cosine":
        return F.cosine_similarity(batch0.unsqueeze(1), batch1.unsqueeze(0), dim=2)
    elif metric == "euclidean":
        distances = torch.cdist(batch0, batch1, p=2)
        similarities = torch.exp(-distances ** 2 / (2 * sigma ** 2))
        return similarities
    elif metric == "soft_cosine":
        if feature_similarity is None:
            raise ValueError("Feature similarity matrix must be provided for soft cosine similarity")
        return soft_cosine_similarity(batch0, batch1, feature_similarity)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def batch_similarity_scores(matrix, metric="cosine", batch_size=256, sigma=1.0):
    feature_similarity = compute_feature_similarity(matrix)
    n = matrix.size(0)
    similarity_matrix = None

    for i in tqdm(range(0, n, batch_size), desc=f"Calculating {metric} similarities"):
        batch0 = matrix[i:i + batch_size]
        batch_similarity_matrix = None

        for j in range(0, n, batch_size):
            batch1 = matrix[j:j + batch_size]

            if batch_similarity_matrix is None:
                batch_similarity_matrix = similarity_scores(batch0, batch1, metric, feature_similarity, sigma)
            else:
                batch_batch_similarity_matrix = similarity_scores(batch0, batch1, metric, feature_similarity, sigma)
                batch_similarity_matrix = torch.cat((batch_similarity_matrix, batch_batch_similarity_matrix), dim=1)

        if similarity_matrix is None:
            similarity_matrix = batch_similarity_matrix
        else:
            if similarity_matrix.size(1) != batch_similarity_matrix.size(1):
                difference = similarity_matrix.size(1) - batch_similarity_matrix.size(1)
                padding = (0, abs(difference)) if difference < 0 else (0, 0)
                batch_similarity_matrix = F.pad(batch_similarity_matrix, padding)
                similarity_matrix = F.pad(similarity_matrix, (-difference, 0))
            similarity_matrix = torch.cat((similarity_matrix, batch_similarity_matrix))

    return similarity_matrix


def get_all_similarities(embeddings, sigma=1.0):
    cosine_similarity = batch_similarity_scores(embeddings, metric="cosine")
    soft_cosine_similarity = batch_similarity_scores(embeddings, metric="soft_cosine")
    euclidean_similarity = batch_similarity_scores(embeddings, metric="euclidean", sigma=sigma)

    return cosine_similarity, soft_cosine_similarity, euclidean_similarity


document_embeddings = torch.randn(100, 300)  # Example document embeddings
feature_similarity = compute_feature_similarity(document_embeddings)

batch0 = document_embeddings[:10]  # First 10 samples
batch1 = document_embeddings[10:20]  # Next 10 samples

cosine_sim, soft_cos_sim, euclidean_sim = get_all_similarities(document_embeddings)
print("Cosine Similarity:\n", cosine_sim)
print("Soft Cosine Similarity:\n", soft_cos_sim)
print("Euclidean Similarity:\n", euclidean_sim)
