import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.cluster import homogeneity_score, completeness_score

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

def get_tags_count(tags):
    flattened_tags = [tag for sublist in tags for tag in sublist]
    tag_counts = Counter(flattened_tags)
    return tag_counts


def group_into_clusters(labels, data):
    unique_labels = np.unique(labels)
    groups = {label: [] for label in unique_labels}
    for label, item in zip(labels, data):
        groups[label].append(item)
    return groups


def get_clusters_with_tags(embeddings, tags, clusterer_f):
    cluster_labels = clusterer_f(embeddings)
    clusters = group_into_clusters(cluster_labels, [tag for sublist in tags for tag in sublist])
    return clusters


def calculate_homogeneity(tags_by_cluster):
    true_labels = []
    predicted_clusters = []

    for cluster_id, tags in tags_by_cluster.items():
        for tag in tags:
            true_labels.append(tag)
            predicted_clusters.append(cluster_id)

    return homogeneity_score(true_labels, predicted_clusters)


def calculate_completeness(tags_by_cluster):
    true_labels = []
    predicted_clusters = []

    for cluster_id, tags in tags_by_cluster.items():
        for tag in tags:
            true_labels.append(tag)
            predicted_clusters.append(cluster_id)

    return completeness_score(true_labels, predicted_clusters)


def calculate_tag_concentration_purity(tags_by_cluster, tag_counts, k):
    """
    For each of the K most popular tags, find the cluster that contains the highest
    number of documents with that tag. Calculate the percentage of all documents 
    with that tag contained within this cluster.
    """
    most_popular_tags = [tag for tag, _ in Counter(tag_counts).most_common(k)]
    
    purity_scores = {}
    
    for tag in most_popular_tags:
        max_cluster_count = 0
        
        for _, tags in tags_by_cluster.items():
            tag_count_in_cluster = tags.count(tag)
            if tag_count_in_cluster > max_cluster_count:
                max_cluster_count = tag_count_in_cluster
        
        if tag in tag_counts and tag_counts[tag] > 0:
            purity_score = max_cluster_count / tag_counts[tag]
        else:
            purity_score = 0  # If no documents have the tag, purity is 0
        
        purity_scores[tag] = purity_score
    
    return purity_scores


def calculate_cluster_tag_purity(tags_by_cluster, tag_counts, total_docs, k):
    """
    For each of the K most popular tags, find the cluster that contains the highest
    number of documents with that tag. Calculate the percentage of documents in this
    cluster that have the tag, over all documents in the dataset.
    """
    most_popular_tags = [tag for tag, _ in Counter(tag_counts).most_common(k)]
    
    purity_scores = {}
    
    for tag in most_popular_tags:
        max_cluster_count = 0
        
        for _, tags in tags_by_cluster.items():
            tag_count_in_cluster = tags.count(tag)
            if tag_count_in_cluster > max_cluster_count:
                max_cluster_count = tag_count_in_cluster
        
        if total_docs > 0:
            purity_score = max_cluster_count / total_docs
        else:
            purity_score = 0  # If there are no documents, purity is 0
        
        purity_scores[tag] = purity_score
    
    return purity_scores









