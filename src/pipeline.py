import embeddings as embed
import similarity as sim
import graph_generation as gg

def compose(*functions):
    def composed_function(data):
        result = data
        for function in functions:
            result = function(result)
        return result
    return composed_function


def make_pipeline(*functions):
    return compose(*functions)


if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt

    data = [(141, "Hello, my name is Erik."), (58, "What is that song called?"),
               (117, "Tell me the name of that song."), (6, "What year was that song made?")]

    ids = [id for id, _ in data]
    strings = [s for _, s in data]
    
    model_id = 'sentence-transformers/all-mpnet-base-v2'
    # model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer, model = embed.initialize_embedding_model(model_id)

    pipeline = make_pipeline(
                     lambda data: embed.batch_embeddings(data, tokenizer, model),
                     sim.batch_similarity_scores,
                     lambda data: gg.knn_graph(data, ids)
                    )
    
    G = pipeline(strings)

    pos = nx.circular_layout(G)  # Define the layout for the nodes
    nx.draw(G, pos, with_labels=True, node_size=700, font_size=10)  # Draw the nodes with labels
    edge_labels = nx.get_edge_attributes(G, 'weight')  # Get edge weights
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)  # Draw edge labels

    # Display the graph
    plt.show()
    
    print(f"Output:\n{G}")
    


