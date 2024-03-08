def compose_pipeline(*functions):
    def composed_function(data):
        result = data
        for function in functions:
            result = function(result)
        return result
    return composed_function


if __name__ == "__main__":
    from utils import pca
    from embeddings import initialize_embedding_model
    from similarity import batch_similarity_scores
    from graph_generation import knn_graph, graph_to_json

    import torch
    import pandas as pd

    # data = [(141, "Hello, my name is Erik."), (58, "What is that song called?"),
    #            (117, "Tell me the name of that song."), (6, "What year was that song made?")]

    # ids = [id for id, _ in data]
    # strings = [s for _, s in data]

    data = pd.read_csv("data/medium_1k_tags_5k_obs.csv")
    data = data.iloc[:100]
    ids = data.index.tolist()
    titles = data["title"].tolist()

    embeddings = torch.load("data/embeddings_1k_tags_5k_obs.pt")
    embeddings = embeddings[:100]

    pca_encodings = [tuple(encoding) for encoding in pca(embeddings, n_components=5)]

    # model_id = 'sentence-transformers/all-mpnet-base-v2'
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer, model = initialize_embedding_model(model_id)

    pipeline = compose_pipeline(
                     batch_similarity_scores,
                     lambda data: knn_graph(data, ids, encodings=pca_encodings, titles=titles, k=3)
                    )

    G = pipeline(embeddings)
    graph_to_json(G, "visualization/graph_data_k3.json")
