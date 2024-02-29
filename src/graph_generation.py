import torch
import networkx as nx

from utils import sort_matrix_values

def knn_graph(matrix: torch.Tensor, document_ids, k=10):
    indices, values = sort_matrix_values(matrix, k)
    indices = indices[:, 1:]
    weights = values[:, 1:]

    G = nx.Graph()
    G.add_nodes_from(document_ids)

    for i in range(len(indices)):
        node0 = document_ids[i]
        for j, w in enumerate(zip(indices[i], weights[i])):
            node1 = document_ids[j]
            G.add_edge(node0, node1, weight=w)

    return G
            


    
