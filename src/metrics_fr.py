import pandas as pd
import numpy as np
from collections import defaultdict

# ------------------------------------- Embeddings

def group_by_tags(tags):
    tags_to_rows = defaultdict(list)
    for i, tags in enumerate(tags):
        for tag in tags:
            tags_to_rows[tag].append(i)
    return tags_to_rows


def get_node_pairs(tag_dict: dict) -> set:
    node_pairs = set()
    for doc_ids in tag_dict.values():
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                node_pairs.add((doc_ids[i], doc_ids[j]))
    return node_pairs


def calculate_metrics(similarity_matrix: np.ndarray, tag_dict: dict, metric_name: str, embedding_model, data_source) -> pd.DataFrame:
    num_nodes = similarity_matrix.shape[0]
    shared_tag_pairs = get_node_pairs(tag_dict)
    upper_tri_indices = np.triu_indices(num_nodes, k=1)

    all_distances = similarity_matrix[upper_tri_indices]
    shared_tag_distances = [similarity_matrix[i, j] for i, j in shared_tag_pairs]
    
    all_distances = np.array(all_distances)
    shared_tag_distances = np.array(shared_tag_distances)

    metrics = {
        'data_source': data_source,
        'embedding_model': embedding_model,
        'metric_name': metric_name,
        'metric': ['mean', 'median', 'std_dev'],
        'between_all_nodes': [
            np.mean(all_distances),
            np.median(all_distances),
            np.std(all_distances)
        ],
        'between_shared_tags': [
            np.mean(shared_tag_distances),
            np.median(shared_tag_distances),
            np.std(shared_tag_distances)
        ]
    }
    
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df


def calculate_embedding_metrics_for_all(cosine_sim, soft_cosine_sim, euclidean_sim, tags, embedding_model: str, data_source: str):
    tag_dict = group_by_tags(tags)

    cosine_metrics_df = calculate_metrics(cosine_sim, tag_dict, 'cosine', embedding_model, data_source)
    soft_cosine_metrics_df = calculate_metrics(soft_cosine_sim, tag_dict, 'soft_cosine', embedding_model, data_source)
    euclidean_metrics_df = calculate_metrics(euclidean_sim, tag_dict, 'euclidean', embedding_model, data_source)

    all_metrics_df = pd.concat([cosine_metrics_df, soft_cosine_metrics_df, euclidean_metrics_df], ignore_index=True)

    return all_metrics_df


# ---------------------------- Clusters


