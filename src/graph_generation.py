import torch
import networkx as nx

from utils import sort_matrix_values


def knn_graph(matrix: torch.Tensor, document_ids, k=10):
    """
    Create a k-nearest neighbors graph based on the input matrix.

    Args:
        matrix (torch.Tensor): Input matrix containing the data points.
        document_ids: IDs of the documents corresponding to the data points.
        k (int): Number of nearest neighbors to consider. Default is 10.
    
    Returns:
        nx.Graph: A networkx graph representing the k-nearest neighbors graph.
    """
    indices, values = sort_matrix_values(matrix, k)
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



            


    
