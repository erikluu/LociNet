import torch
import networkx as nx

from utils import sort_matrix_values


def knn_graph(sim_mat: torch.Tensor, document_ids: list[int], k: int = 10):
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
    print(indices)
    print(weights)
    print(document_ids)

    G = nx.Graph()
    G.add_nodes_from(document_ids)

    for i in range(len(indices)):
        node0 = document_ids[i]
        for j, w in zip(indices[i], weights[i]):
            node1 = document_ids[j]
            print(f"connecting {node0} and {node1}")
            G.add_edge(node0, node1, weight=w)

    return G


def small_world_graph():
    """
    https://chih-ling-hsu.github.io/2020/05/15/watts-strogatz
    """
    pass


if __name__ == "__main__":
    import torch
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt

    import similarity as sim
    import pipeline as pipe

    data = pd.read_csv("../playgrounds/data_medium_1k_tags_5k_obs.csv")
    data = data.iloc[:100]
    ids = data.index.tolist()

    embeddings = torch.load("../data/embeddings_1k_tags_5k_obs.pt")
    embeddings = embeddings[:100]

    pipeline = pipe.compose_pipeline(
        sim.batch_similarity_scores,
        lambda sim_mat: knn_graph(sim_mat, ids, k=5)
    )

    G = pipeline(embeddings)

    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()

    print(f"Output:\n{G}")
