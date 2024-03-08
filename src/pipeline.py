import embeddings as embed
import similarity as sim
import graph_generation as gg
import utils

def compose_pipeline(*functions):
    def composed_function(data):
        result = data
        for function in functions:
            result = function(result)
        return result
    return composed_function


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

    embeddings = embed.batch_embeddings(strings, tokenizer, model)
    pca_encodings = utils.pca(embeddings, n_components=2)

    pipeline = compose_pipeline(
                     sim.batch_similarity_scores,
                     lambda data: gg.knn_graph(data, ids, positions=pca_encodings)
                    )

    G = pipeline(embeddings)
    print(f"Output:\n{G}")
