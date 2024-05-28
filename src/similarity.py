import torch
import torch.nn.functional as F
from tqdm import tqdm
import taichi as ti
import numpy as np

# Initialize Taichi with the Metal backend
ti.init(arch=ti.metal)

# Define Taichi fields dynamically based on the embedding size
def initialize_taichi_fields(embedding_size, batch_size):
    batch0_ti = ti.field(dtype=ti.f32, shape=(batch_size, embedding_size))
    batch1_ti = ti.field(dtype=ti.f32, shape=(batch_size, embedding_size))
    feature_similarity_ti = ti.field(dtype=ti.f32, shape=(embedding_size, embedding_size))
    similarity_matrix_ti = ti.field(dtype=ti.f32, shape=(batch_size, batch_size))
    return batch0_ti, batch1_ti, feature_similarity_ti, similarity_matrix_ti

@ti.kernel
def compute_feature_similarity_ti(features: ti.types.ndarray(), feature_similarity: ti.types.ndarray()):
    for i in range(feature_similarity.shape[0]):
        for j in range(feature_similarity.shape[1]):
            feature_similarity[i, j] = 0.0
            for k in range(features.shape[1]):
                feature_similarity[i, j] += features[k, i] * features[k, j]

@ti.kernel
def soft_cosine_similarity_ti(batch0: ti.types.ndarray(), batch1: ti.types.ndarray(), feature_similarity: ti.types.ndarray(), result: ti.types.ndarray()):
    for i in range(batch0.shape[0]):
        for j in range(batch1.shape[0]):
            numerator = 0.0
            denominator0 = 0.0
            denominator1 = 0.0
            for k in range(batch0.shape[1]):
                for l in range(batch1.shape[1]):
                    numerator += batch0[i, k] * feature_similarity[k, l] * batch1[j, l]
                    denominator0 += batch0[i, k] * feature_similarity[k, l] * batch0[i, l]
                    denominator1 += batch1[j, k] * feature_similarity[k, l] * batch1[j, l]
            result[i, j] = numerator / ti.sqrt(denominator0 * denominator1)

def compute_feature_similarity(features):
    features = features.t()  # Transpose the feature matrix to get features as rows
    features_np = features.cpu().numpy()
    embedding_size = features_np.shape[0]
    feature_similarity_np = np.zeros((embedding_size, embedding_size), dtype=np.float32)
    
    batch0_ti, batch1_ti, feature_similarity_ti, similarity_matrix_ti = initialize_taichi_fields(embedding_size, batch_size=features_np.shape[1])
    compute_feature_similarity_ti(features_np, feature_similarity_np)
    return torch.from_numpy(feature_similarity_np).to(features.device)

def soft_cosine_similarity(batch0, batch1, feature_similarity):
    assert feature_similarity.size(0) == batch0.size(1) and feature_similarity.size(1) == batch1.size(1), \
        "Feature similarity matrix dimensions do not match the feature dimensions of the batches"
    
    norm_batch0 = F.normalize(batch0, p=2, dim=1)
    norm_batch1 = F.normalize(batch1, p=2, dim=1)

    batch0_np = norm_batch0.cpu().numpy()
    batch1_np = norm_batch1.cpu().numpy()
    feature_similarity_np = feature_similarity.cpu().numpy()
    result_np = np.zeros((batch0.size(0), batch1.size(0)), dtype=np.float32)
    
    batch0_ti, batch1_ti, feature_similarity_ti, similarity_matrix_ti = initialize_taichi_fields(batch0.size(1), batch0.size(0))
    soft_cosine_similarity_ti(batch0_np, batch1_np, feature_similarity_np, result_np)
    return torch.from_numpy(result_np).to(batch0.device)

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

# Example usage:
# document_embeddings = torch.randn(100, 300)  # Example document embeddings
# cosine_sim, soft_cos_sim, euclidean_sim = get_all_similarities(document_embeddings)
# print("Cosine Similarity:\n", cosine_sim)
# print("Soft Cosine Similarity:\n", soft_cos_sim)
# print("Euclidean Similarity:\n", euclidean_sim)
