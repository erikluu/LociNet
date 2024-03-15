import json
import torch
import networkx as nx

from utils import sort_matrix_values, normalize_and_convert_to_hex

def graph_to_json(G, save_path: str = "graph_data.json"):
    graph_data = {
        "nodes": [],
        "links": []
    }

    for node, data in G.nodes(data=True):
        print(node, data)
        graph_data["nodes"].append({"id": node, "pos": data["pos"], "color": data["color"], "title": data["title"]})

    for source, target, data in G.edges(data=True):
        graph_data["links"].append({"source": source, "target": target, "weight": data["weight"]})

    with open(save_path, 'w') as json_file:
        json.dump(graph_data, json_file)


def insert_nodes(G, document_ids: list[int], attributes: dict):
    """
    Insert nodes into the graph with the given attributes.

    Args:
        G (nx.Graph): The graph to insert the nodes into.
        document_ids (list[int]): List of document ids.
        attributes (dict): Dictionary of attributes to be added to the nodes.
    Returns:
        None
    """
    nodes = []
    for i, id in enumerate(document_ids):
        attr = {k: v[i] for k, v in attributes.items()}
        nodes.append((i, attr))

    G.add_nodes_from(nodes)


def knn_graph(sim_mat: torch.Tensor, document_ids: list[int], k: int = 10, **kwargs):
    """
    Create a k-nearest neighbors graph based on the given similaity matrix

    Args:
        sim_mat (torch.Tensor): Input matrix the similarity scores between each document.
        k (int): Number of nearest neighbors to consider. Default is 10.
        **kwargs: Additional attributes to be passed to the graph nodes.
    Returns:
        nx.Graph: A networkx graph representing the k-nearest neighbors graph.
    """
    G = nx.Graph()
    insert_nodes(G, document_ids, kwargs)

    indices, values = sort_matrix_values(sim_mat, k+1) # not including itself
    indices = indices[:, 1:]
    weights = values[:, 1:]

    edges = []
    for i in range(len(indices)):
        node0 = document_ids[i]
        for j, weight in zip(indices[i], weights[i]):
            node1 = document_ids[j]
            edges.append((node0, node1, weight.item()))

    G.add_weighted_edges_from(edges)
    return G


def mst_graph(sim_mat: torch.Tensor, document_ids: list[int], **kwargs):
    """
    Create a minimum spanning tree graph based on the given similarity matrix.

    Args:
        sim_mat (torch.Tensor): Input matrix the similarity scores between each document.
        **kwargs: Additional attributes to be passed to the graph nodes.
    Returns:
        nx.Graph: A networkx graph representing the minimum spanning tree graph.
    """
    G = nx.Graph()
    insert_nodes(G, document_ids, kwargs)

    indices, values = sort_matrix_values(sim_mat, len(document_ids)-1)
    edges = []
    for i in range(len(indices)):
        node0 = document_ids[i]
        for j, weight in zip(indices[i], values[i]):
            node1 = document_ids[j]
            edges.append((node0, node1, weight.item()))

    G.add_weighted_edges_from(edges)
    return nx.minimum_spanning_tree(G)


def dbscan_graph(clusters: torch.Tensor, document_ids: list[int], **kwargs):
    """
    Create a graph based on the DBSCAN clustering algorithm.

    Args:
        sim_mat (torch.Tensor): Input matrix the similarity scores between each document.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        **kwargs: Additional attributes to be passed to the graph nodes.
    Returns:
        nx.Graph: A networkx graph representing the DBSCAN clustering graph.
    """
    G = nx.Graph()
    insert_nodes(G, document_ids, kwargs)

    X = sim_mat.numpy()




def small_world_graph():
    """
    https://chih-ling-hsu.github.io/2020/05/15/watts-strogatz
    """
    pass


if __name__ == "__main__":
    sim_mat = torch.randn(10, 10)

    ids = list(range(10))
    positions = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
    colors = ["black", "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "grey"]
    titles = ["title"] * 10

    G = knn_graph(sim_mat, k=3, document_ids=ids, positions=positions, colors=colors, titles=titles)
    graph_to_json(G)
