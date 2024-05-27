import random
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import homogeneity_score, completeness_score
import matplotlib.pyplot as plt
import seaborn as sns


import src.utils as utils

# ------------------------------------------------------------
# -------------------------- Helpers -------------------------
# ------------------------------------------------------------
def get_all_cluster_ids(graph):
   return [node[1]["cluster_id"] for node in graph.nodes(data=True) if node[1]["titles"].startswith("ClusterNode_")]


def get_n_unique_tags(graph, n):
    all_tags = list({node["tags"] for node in graph.nodes(data=True) if node.startswith("ClusterNode_")})
    assert len(all_tags) >= n
    return np.random.choice(all_tags, size=n, replace=False)


def metric_for_each_cluster(metric_f, graph, cluster_ids):
    metric_by_cluster = {}
    for id in cluster_ids:
        metric_by_cluster[id] = metric_f(graph, id)
    
    return metric_by_cluster


def aggregate_metrics(graph, sample_size=10, depth=2, n_tags=10):
    np.random.seed(42)
    cluster_ids = get_all_cluster_ids(graph)

    avg_embedding_dist_within_cluster = metric_for_each_cluster(average_embedding_distance_within_cluster, graph, cluster_ids)
    avg_embedding_dist_overall = average_embedding_distance_overall(graph)
    avg_edge_dist_within_cluster = metric_for_each_cluster(average_edge_distance_within_cluster, graph, cluster_ids)
    avg_dist_to_same_tags = average_distance_to_nodes_with_same_tags(graph, sample_size)
    tag_variation_bfs_result = tag_variation_bfs(graph, sample_size, depth)

    tag_overlap_result = tag_overlap_summary(graph, get_n_unique_tags(graph, n_tags))

    metrics = {
        'Metric': [
            'Average Embedding Distance Within Cluster',
            'Average Embedding Distance Overall',
            'Average Edge Distance Within Cluster (Weighted)',
            'Average Distance to Same Tags',
            'Tag Variation in BFS',
            'Tag Overlap'
        ],
        'Result': [
            avg_embedding_dist_within_cluster,
            avg_embedding_dist_overall,
            avg_edge_dist_within_cluster,
            avg_dist_to_same_tags,
            tag_variation_bfs_result,
            tag_overlap_result
        ]
    }

    metrics_df = pd.DataFrame(metrics)
    return metrics_df


# ------------------------------------------------------------
# --------------------- Embedding Testing --------------------
# ------------------------------------------------------------
def group_by_tags(tags):
    tags_to_rows = defaultdict(list)
    for i, tags in enumerate(tags):
        for tag in tags:
            tags_to_rows[tag].append(i)
    return tags_to_rows


def get_node_pairs(tag_dict: dict) -> set:
    """Get pairs of nodes that share one or more tags.
    
    Args:
        tag_dict (dict): Dictionary where keys are tags and values are lists of document IDs.
        
    Returns:
        set: A set of tuples representing pairs of nodes that share tags.
    """
    node_pairs = set()
    for doc_ids in tag_dict.values():
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                node_pairs.add((doc_ids[i], doc_ids[j]))
    return node_pairs


def calculate_metrics(similarity_matrix: np.ndarray, tag_dict: dict, metric_name: str, embedding_model, data_source) -> pd.DataFrame:
    """Calculate and organize metrics for a similarity matrix using a tag dictionary.
    
    Args:
        similarity_matrix (np.ndarray): A square similarity matrix.
        tag_dict (dict): Dictionary where keys are tags and values are lists of document IDs.
        metric_name (str): Name of the distance metric used.
        
    Returns:
        pd.DataFrame: DataFrame with organized metrics.
    """
    # Get the number of nodes
    num_nodes = similarity_matrix.shape[0]
    
    # Get pairs of nodes that share one or more tags
    shared_tag_pairs = get_node_pairs(tag_dict)
    
    # Calculate upper triangular indices, excluding the diagonal
    upper_tri_indices = np.triu_indices(num_nodes, k=1)
    
    # Extract all distances from the upper triangular part of the matrix
    all_distances = similarity_matrix[upper_tri_indices]
    
    # Extract distances for pairs of nodes that share tags
    shared_tag_distances = [similarity_matrix[i, j] for i, j in shared_tag_pairs]
    
    # Convert to numpy arrays for statistics calculation
    all_distances = np.array(all_distances)
    shared_tag_distances = np.array(shared_tag_distances)

    # Calculate statistics
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
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df


def calculate_embedding_metrics_for_all(cosine_sim, soft_cosine_sim, euclidean_sim, tags, embedding_model: str, data_source: str):
    """Calculate metrics for all distance metrics and combine them into a single DataFrame.
    
    Args:
        cosine_sim (np.ndarray): Cosine similarity matrix.
        soft_cosine_sim (np.ndarray): Soft cosine similarity matrix.
        euclidean_sim (np.ndarray): Euclidean similarity matrix.
        tags (list): List of tags for the documents.
        group_by_tags_func (function): Function to group documents by tags.
        
    Returns:
        pd.DataFrame: Combined DataFrame with metrics for all distance metrics.
    """
    tag_dict = group_by_tags(tags)

    cosine_metrics_df = calculate_metrics(cosine_sim, tag_dict, 'cosine', embedding_model, data_source)
    soft_cosine_metrics_df = calculate_metrics(soft_cosine_sim, tag_dict, 'soft_cosine', embedding_model, data_source)
    euclidean_metrics_df = calculate_metrics(euclidean_sim, tag_dict, 'euclidean', embedding_model, data_source)

    all_metrics_df = pd.concat([cosine_metrics_df, soft_cosine_metrics_df, euclidean_metrics_df], ignore_index=True)

    return all_metrics_df


def visualize_embeddings_comparisons(dataframes: list[pd.DataFrame]):
    """
    Visualizes comparisons between all nodes and shared tags for multiple dataframes.

    Args:
        dataframes (List[pd.DataFrame]): A list of dataframes to be combined and visualized.
    """
    # Combine the dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Melt the dataframe to have 'between_all_nodes' and 'between_shared_tags' in a single column
    melted_df = combined_df.melt(id_vars=['data_source', 'embedding_model', 'metric_name', 'metric'], 
                                 value_vars=['between_all_nodes', 'between_shared_tags'], 
                                 var_name='comparison_type', value_name='value')

    # Set up the matplotlib figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    # Bar plot for mean, median, std_dev comparison
    sns.barplot(data=melted_df[melted_df['metric'] == 'mean'], x='metric_name', y='value', hue='comparison_type', errorbar=None, ax=axes[0])
    axes[0].set_title('Mean Comparison Between All Nodes and Shared Tags')
    axes[0].set_xlabel('Metric Name')
    axes[0].set_ylabel('Value')
    axes[0].legend(title='Comparison Type')

    sns.barplot(data=melted_df[melted_df['metric'] == 'median'], x='metric_name', y='value', hue='comparison_type', errorbar=None, ax=axes[1])
    axes[1].set_title('Median Comparison Between All Nodes and Shared Tags')
    axes[1].set_xlabel('Metric Name')
    axes[1].set_ylabel('Value')
    axes[1].legend(title='Comparison Type')

    sns.barplot(data=melted_df[melted_df['metric'] == 'std_dev'], x='metric_name', y='value', hue='comparison_type', errorbar=None, ax=axes[2])
    axes[2].set_title('Standard Deviation Comparison Between All Nodes and Shared Tags')
    axes[2].set_xlabel('Metric Name')
    axes[2].set_ylabel('Value')
    axes[2].legend(title='Comparison Type')

    # Adjust layout
    plt.tight_layout()
    plt.show()
    # Create separate plots for each embedding model for clarity
    for model in combined_df['embedding_model'].unique():
        model_df = melted_df[melted_df['embedding_model'] == model]
        
        # Line plot for each model
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=model_df, x='metric_name', y='value', hue='comparison_type', style='metric', markers=True, errorbar=None)
        plt.title(f'Metric Trends Comparison for {model}')
        plt.xlabel('Metric Name')
        plt.ylabel('Value')
        plt.legend(title='Comparison Type')
        plt.show()

# ------------------------------------------------------------
# ---------------------- Cluster Testing ---------------------
# ------------------------------------------------------------

def get_tags_count(tags):
    flattened_tags = [tag for sublist in tags for tag in sublist]
    tag_counts = Counter(flattened_tags)
    return tag_counts


def get_clusters_with_tags(embeddings, tags, clusterer_f):
    cluster_labels = clusterer_f(embeddings)
    clusters = utils.group_into_clusters(cluster_labels, [tag for sublist in tags for tag in sublist])
    return clusters


def calculate_homogeneity(cluster_tags, total_tag_counts):
    # Flatten the cluster tags into a list of true labels and predicted clusters
    true_labels = []
    predicted_clusters = []

    for cluster_id, tags in cluster_tags.items():
        for tag in tags:
            true_labels.append(tag)
            predicted_clusters.append(cluster_id)

    return homogeneity_score(true_labels, predicted_clusters)


def calculate_completeness(cluster_tags, total_tag_counts):
    # Flatten the cluster tags into a list of true labels and predicted clusters
    true_labels = []
    predicted_clusters = []

    for cluster_id, tags in cluster_tags.items():
        for tag in tags:
            true_labels.append(tag)
            predicted_clusters.append(cluster_id)

    return completeness_score(true_labels, predicted_clusters)


def calculate_cluster_purity_for_popular_tags(cluster_tags, total_tag_counts, K):
    # Identify the K most popular tags
    most_popular_tags = [tag for tag, count in Counter(total_tag_counts).most_common(K)]
    
    purity_scores = {}
    
    for tag in most_popular_tags:
        max_cluster_count = 0
        max_cluster_id = None
        
        for cluster_id, tags in cluster_tags.items():
            tag_count_in_cluster = tags.count(tag)
            if tag_count_in_cluster > max_cluster_count:
                max_cluster_count = tag_count_in_cluster
                max_cluster_id = cluster_id
        
        if max_cluster_id is not None:
            purity_score = max_cluster_count / total_tag_counts[tag]
            purity_scores[tag] = purity_score
    
    return purity_scores


def calculate_metrics_dataframe(cluster_tags, total_tag_counts, k=2):
    homogeneity = calculate_homogeneity(cluster_tags, total_tag_counts)
    completeness = calculate_completeness(cluster_tags, total_tag_counts)
    purity_scores = calculate_cluster_purity_for_popular_tags(cluster_tags, total_tag_counts, k)

    # Convert purity_scores to a list of dictionaries for the DataFrame
    purity_scores_list = [{"Tag": tag, "Purity": purity} for tag, purity in purity_scores.items()]

    # Create the DataFrame
    metrics_df = pd.DataFrame(purity_scores_list)
    metrics_df["Homogeneity"] = homogeneity
    metrics_df["Completeness"] = completeness

    return metrics_df

# ------------------------------------------------------------
# ----------------------- Graph Testing ----------------------
# ------------------------------------------------------------

# degree of separation ---------------------------------------
def calculate_degree_of_separation_metrics(G):
    # Compute shortest path lengths between all pairs of nodes
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    # Extract the lengths into a list, excluding self-loops
    lengths = []
    for source, targets in path_lengths.items():
        for target, length in targets.items():
            if source != target:  # Exclude self-loops
                lengths.append(length)
    
    # Calculate statistical metrics
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    mode_length = stats.mode(lengths).mode[0]
    percentiles = np.percentile(lengths, [25, 50, 75])
    
    metrics = {
        'mean': mean_length,
        'median': median_length,
        'mode': mode_length,
        '25th_percentile': percentiles[0],
        '50th_percentile': percentiles[1],  # Same as median
        '75th_percentile': percentiles[2]
    }

    return metrics


def calculate_degree_of_separation_within_clusters(G):
    clusters = {}
    for node, data in G.nodes(data=True):
        cluster_label = data['cluster_id']
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(node)
    
    cluster_metrics = {}
    
    # Compute metrics for each cluster
    for cluster_label, nodes in clusters.items():
        subgraph = G.subgraph(nodes)
        path_lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
        
        lengths = []
        for source, targets in path_lengths.items():
            for target, length in targets.items():
                if source != target:  # Exclude self-loops
                    lengths.append(length)
        
        if lengths:
            mean_length = np.mean(lengths)
            median_length = np.median(lengths)
            mode_length = stats.mode(lengths).mode[0]  # Most common length
            percentiles = np.percentile(lengths, [25, 50, 75])  # 25th, 50th (median), 75th percentiles
            
            cluster_metrics[cluster_label] = {
                'mean': mean_length,
                'median': median_length,
                'mode': mode_length,
                '25th_percentile': percentiles[0],
                '50th_percentile': percentiles[1],  # Same as median
                '75th_percentile': percentiles[2]
            }
        else:
            cluster_metrics[cluster_label] = {
                'mean': None,
                'median': None,
                'mode': None,
                '25th_percentile': None,
                '50th_percentile': None,
                '75th_percentile': None
            }
    
    return cluster_metrics


def calculate_degree_of_separation_for_shared_tags(G):
    tag_to_nodes = {}
    for node, data in G.nodes(data=True):
        for tag in data['simplified_tags']:
            if tag not in tag_to_nodes:
                tag_to_nodes[tag] = set()
            tag_to_nodes[tag].add(node)
    
    # Combine nodes sharing any tag into one set
    nodes_with_shared_tags = set()
    for nodes in tag_to_nodes.values():
        nodes_with_shared_tags.update(nodes)
    
    subgraph = G.subgraph(nodes_with_shared_tags)
    path_lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
    
    lengths = []
    for source, targets in path_lengths.items():
        for target, length in targets.items():
            if source != target:  # Exclude self-loops
                lengths.append(length)
    
    if lengths:
        mean_length = np.mean(lengths)
        median_length = np.median(lengths)
        mode_length = stats.mode(lengths).mode[0]  # Most common length
        percentiles = np.percentile(lengths, [25, 50, 75])  # 25th, 50th (median), 75th percentiles
        
        metrics = {
            'mean': mean_length,
            'median': median_length,
            'mode': mode_length,
            '25th_percentile': percentiles[0],
            '50th_percentile': percentiles[1],  # Same as median
            '75th_percentile': percentiles[2]
        }
    else:
        metrics = {
            'mean': None,
            'median': None,
            'mode': None,
            '25th_percentile': None,
            '50th_percentile': None,
            '75th_percentile': None
        }
    
    return metrics

# rando stuff ------------------------------------------
def average_embedding_distance_within_cluster(graph, cluster_id):
    cluster_embeddings = []
    for _, attributes in graph.nodes(data=True):
        if attributes.get('cluster_assignment') == cluster_id:
            cluster_embeddings.append(attributes['embeddings'])

    if len(cluster_embeddings) < 2:
        return np.nan  # Not enough nodes to compute distance

    cluster_embeddings_array = np.stack(cluster_embeddings)
    pairwise_distances = 1 - cosine_similarity(cluster_embeddings_array)
    average_distance = np.mean(pairwise_distances[np.triu_indices(pairwise_distances.shape[0], k=1)])
    return average_distance


def average_embedding_distance_overall(graph):
    all_embeddings = []
    for _, attributes in graph.nodes(data=True):
        all_embeddings.append(attributes['embeddings'])

    if len(all_embeddings) < 2:
        return np.nan  # Not enough nodes to compute distance

    all_embeddings_array = np.stack(all_embeddings)
    pairwise_distances = 1 - cosine_similarity(all_embeddings_array)
    average_distance = np.mean(pairwise_distances[np.triu_indices(pairwise_distances.shape[0], k=1)])
    return average_distance


def average_edge_distance_overall(graph):
    edge_distances = []
    for edge in graph.edges(data=True):
        _, _, edge_data = edge
        edge_weight = edge_data.get('weight', 1)  # Default weight is 1 if not specified
        edge_distances.append(edge_weight)

    if len(edge_distances) == 0:
        return np.nan  # No edges

    average_distance = np.mean(edge_distances)
    return average_distance


def average_edge_distance_within_cluster(graph, cluster_id):
    edge_distances = []
    for edge in graph.edges(data=True):
        source_node, target_node, edge_data = edge
        source_cluster = graph.nodes[source_node].get('cluster_assignment')
        target_cluster = graph.nodes[target_node].get('cluster_assignment')
        # Check if both source and target nodes belong to the specified cluster
        if source_cluster == cluster_id and target_cluster == cluster_id:
            edge_weight = edge_data.get('weight', 1)  # Default weight is 1 if not specified
            edge_distances.append(edge_weight)

    if len(edge_distances) == 0:
        return np.nan  # No edges within the cluster

    average_distance = np.mean(edge_distances)
    return average_distance


def average_distance_to_nodes_with_same_tags(graph, n_nodes):
    nodes = list(graph.nodes(data=True))
    sampled_nodes = random.sample(nodes, n_nodes)
    
    all_distances = []
    
    for node_id, node_data in sampled_nodes:
        same_tag_nodes = []
        
        # Find all nodes that share at least one tag with the sampled node
        node_tags = set(node_data.get('tags', []))
        for other_node_id, other_node_data in nodes:
            if other_node_id == node_id:
                continue
            other_node_tags = set(other_node_data.get('tags', []))
            if node_tags & other_node_tags:
                same_tag_nodes.append(other_node_id)
        
        # Calculate the distances from the sampled node to nodes with the same tags
        if same_tag_nodes:
            distances = []
            for other_node_id in same_tag_nodes:
                try:
                    distance = nx.shortest_path_length(graph, source=node_id, target=other_node_id)
                    distances.append(distance)
                except nx.NetworkXNoPath:
                    continue
            if distances:
                avg_distance = np.mean(distances)
                all_distances.append(avg_distance)
    
    if all_distances:
        overall_avg_distance = np.mean(all_distances)
        return overall_avg_distance
    
    return np.nan


def tag_variation_bfs(graph, n_nodes, depth):
    nodes = list(graph.nodes(data=True))
    sampled_nodes = random.sample(nodes, n_nodes)
    
    tag_variations = []
    
    for node_id, node_data in sampled_nodes:
        visited_tags = []
        
        # Perform BFS up to a specified depth
        for current_node, current_depth in nx.bfs_edges(graph, source=node_id, depth_limit=depth):
            if current_depth <= depth:
                current_node_data = graph.nodes[current_node]
                visited_tags.extend(current_node_data.get('tags', []))
        
        # Calculate tag variation
        tag_counts = Counter(visited_tags)
        total_tags = sum(tag_counts.values())
        if total_tags > 0:
            tag_variation = {tag: count / total_tags for tag, count in tag_counts.items()}
            tag_variations.append(tag_variation)
    
    return tag_variations


def tag_overlap_summary(graph, tags):
    tag_overlap_d = {}
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            tag_overlap_d[f"{tags[i]} {tags[j]}"] = tag_overlap(graph, tags[i], tags[j])
    
    return tag_overlap_d


def tag_overlap(graph, tag1, tag2):
    "Just by nodes individual nodes, not subgraph overlap. I'm sure this works too..."
    nodes_with_tag1 = {node for node, data in graph.nodes(data=True) if tag1 in data.get('tags', [])}
    nodes_with_tag2 = {node for node, data in graph.nodes(data=True) if tag2 in data.get('tags', [])}
    
    intersection = nodes_with_tag1.intersection(nodes_with_tag2)
    union = nodes_with_tag1.union(nodes_with_tag2)
    
    if not union:  # Avoid division by zero if both sets are empty
        return 0.0
    
    overlap_measure = len(intersection) / len(union)
    
    return overlap_measure


# -------------------------------------------------------------
# ------------ NetworkX Built-in General Metrics --------------
# -------------------------------------------------------------
def networkx_metrics(G, remove_cluster_nodes=False):
    if remove_cluster_nodes:
        cluster_node_ids = []
        for node_id in G.nodes():
            if "ClusterNode_" in str(node_id):
                cluster_node_ids.append(node_id)

        G.remove_nodes_from(cluster_node_ids)                

    pdd = degree_distribution(G)
    pdd["clustering_coefficient"] = clustering_coefficient(G)

    df = pd.DataFrame()
    df["degree_centrality"] = degree_centrality(G)
    df["between_centrality"] = betweenness_centrality(G)
    df["closeness_centrality"] = closeness_centrality(G)
    df["assortativity_coefficient"] = assortativity_coefficient(G)
    df["global_efficiency"] = global_efficiency(G)
    df["local_efficiency"] = local_efficiency(G)
    df["largest_connected_component_size"] = largest_connected_component_size(G)

    if not remove_cluster_nodes: # need connected graphs
        pdd["average_path_length"] = average_path_length(G)
        df["detect_communities"] = detect_communities(G)

    return pdd, df


def degree_distribution(G):
    """
    Description: Degree distribution refers to the distribution of node degrees (i.e., the number of connections each node has) across the graph. It describes how many nodes have a given degree.

    Insights: Understanding the degree distribution can reveal whether the graph follows a specific pattern, such as a power-law distribution indicative of scale-free networks, or a bell-shaped distribution typical of random networks. It provides insights into the connectivity and heterogeneity of the graph.
    """
    degrees = np.array([degree for node, degree in G.degree()])
    return {
        "min": min(degrees),
        "max": max(degrees),
        "mean": np.mean(degrees),
        "median": np.median(degrees),
        "std_dev": np.std(degrees),
        "q1": np.percentile(degrees, 25),
        "q3": np.percentile(degrees, 75)
    }
    # plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), density=True)
    # plt.title("Degree Distribution")
    # plt.xlabel("Degree")
    # plt.ylabel("Probability")
    # plt.show()


def clustering_coefficient(G):
    """
    Description: The clustering coefficient measures the extent to which nodes in the graph tend to cluster together. It quantifies the probability that two nodes connected to a common neighbor are also connected to each other.

    Insights: A high clustering coefficient suggests a network with many tightly connected clusters, indicating a high level of local connectivity and potential community structure. It reflects the degree of transitivity or 'cliquishness' in the graph.
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.average_clustering.html
    """
    return nx.average_clustering(G)


def average_path_length(G):
    """
    Description: Average path length measures the average shortest path length between all pairs of nodes in the graph. It quantifies how efficiently information can spread through the network.

    Insights: A shorter average path length indicates greater overall connectivity and easier information flow between nodes. It reflects the global efficiency of the network.
    """
    return nx.average_shortest_path_length(G)


def degree_centrality(G):
    """
    Description: Degree centrality measures the importance of a node in the graph based on the number of connections it has. Nodes with higher degree centrality are more central to the network structure.

    Insights: Degree centrality identifies the most connected nodes in the network. Nodes with high degree centrality may serve as important hubs or connectors within the network.
    """
    return nx.degree_centrality(G)


def betweenness_centrality(G):
    """
    Description: Betweenness centrality quantifies the extent to which a node lies on the shortest paths between other nodes. It measures the node's importance in facilitating communication between other nodes.

    Insights: Nodes with high betweenness centrality act as bridges or bottlenecks in the network, controlling the flow of information between different parts of the network. They are critical for maintaining communication and connectivity.
    """
    return nx.betweenness_centrality(G)


def closeness_centrality(G):
    """
    Description: Closeness centrality measures how close a node is to all other nodes in the graph. It quantifies how quickly a node can interact with other nodes in the network.

    Insights: Nodes with high closeness centrality are able to reach other nodes in the network more quickly, making them efficient communicators or influencers within the network.
    """
    return nx.closeness_centrality(G)


def detect_communities(G):
    """
    Description: Community structure detection identifies groups of nodes that are densely connected within themselves but sparsely connected to nodes in other groups. It reveals the presence of cohesive subgroups or communities in the network.

    Insights: Community structure analysis uncovers the modular organization of the network, highlighting groups of nodes with shared characteristics or functions. It can help identify specialized roles or functional modules within the network.
    """

    return nx.algorithms.community.greedy_modularity_communities(G)


def assortativity_coefficient(G):
    """
    Description: Assortativity coefficient measures the degree correlation between connected nodes. It quantifies whether nodes tend to connect to other nodes with similar degrees (positive assortativity) or dissimilar degrees (negative assortativity).

    Insights: Assortativity coefficient reveals the mixing pattern of nodes in the network. Positive assortativity indicates the presence of hubs connecting to other hubs, while negative assortativity suggests a more heterogeneous connectivity pattern.
    """
    return nx.degree_assortativity_coefficient(G)


def global_efficiency(G):
    """
    Description: Network efficiency measures how efficiently information can be transferred within the network. Global efficiency quantifies the average efficiency of information exchange across all pairs of nodes, while local efficiency focuses on efficiency within neighborhoods of nodes.

    Insights: Network efficiency metrics provide insights into how easily information can spread through the network and how well different parts of the network are interconnected.
    """
    return nx.global_efficiency(G)


def local_efficiency(G):
    return nx.local_efficiency(G)


def largest_connected_component_size(G):
    """
    Description: Network resilience measures the robustness of the network to node or edge failures. It quantifies the network's ability to maintain connectivity and functionality under perturbations.

    Insights: Network resilience analysis helps identify critical nodes or edges whose removal could significantly impact the network's structure or function. It assesses the network's vulnerability to disruptions and informs strategies for improving resilience.
    """
    return len(max(nx.connected_components(G), key=len))
