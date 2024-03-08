import json
import torch
import networkx as nx

from utils import sort_matrix_values

def graph_to_json(G):
    graph_data = {
        "nodes": [],
        "links": []
    }

    for node, data in G.nodes(data=True):
        print(node, data)
        graph_data["nodes"].append({"id": node, "pos": data["pos"]})

    for source, target, data in G.edges(data=True):
        graph_data["links"].append({"source": source, "target": target, "weight": data["weight"]})

    # Save graph data as JSON
    with open('graph_data.json', 'w') as json_file:
        json.dump(graph_data, json_file)


def knn_graph(sim_mat: torch.Tensor, document_ids: list[int], positions: list[tuple], k: int = 10):
    """
    Create a k-nearest neighbors graph based on the given similaity matrix

    Args:
        sim_mat (torch.Tensor): Input matrix the similarity scores between each document.
        document_ids: IDs of the documents corresponding to the data points.
        k (int): Number of nearest neighbors to consider. Default is 10.

    Returns:
        nx.Graph: A networkx graph representing the k-nearest neighbors graph.
    """
    indices, values = sort_matrix_values(sim_mat, k)
    indices = indices[:, 1:]
    weights = values[:, 1:]

    G = nx.Graph()
    for i, pos in zip(document_ids, positions):
        G.add_node(i, pos=pos)

    edges = []
    for i in range(len(indices)):
        node0 = document_ids[i]
        for j, w in zip(indices[i], weights[i]):
            node1 = document_ids[j]
            edges.append((node0, node1, w.item()))

    G.add_weighted_edges_from(edges)
    return G


def small_world_graph():
    """
    https://chih-ling-hsu.github.io/2020/05/15/watts-strogatz
    """
    pass


if __name__ == "__main__":
    sim_mat = torch.randn(10, 10)
    ids = list(range(10))
    positions = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
    G = knn_graph(sim_mat, ids, positions, k=3)
    graph_to_json(G)
