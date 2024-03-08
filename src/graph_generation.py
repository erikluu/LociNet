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


def insert_nodes(G: nx.Graph, document_ids: list[int], attributes: dict):
    pass
    # nodes = []
    # for id, attr in zip(document_ids, attributes):
    #     node = [id]

    #     for k, v in attr.items():
    #         node.append({k: v})


    # print(nodes)


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
    indices, values = sort_matrix_values(sim_mat, k+1) # not including itself
    indices = indices[:, 1:]
    weights = values[:, 1:]

    G = nx.Graph()
    G = insert_nodes(G, document_ids, kwargs)
    print(G)
    exit()
    # for i, encoding, title in zip(document_ids, encodings, titles):
    #     G.add_node(i, pos=encoding[:2], color=normalize_and_convert_to_hex(encoding[2:5]), title=title)

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
    colors = ["black", "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "grey"]
    titles = ["title"] * 10

    G = knn_graph(sim_mat, k=3, document_ids=ids, positions=positions, colors=colors, titles=titles)
    graph_to_json(G)
