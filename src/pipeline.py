import embeddings as embed
import similarity as sim
import clustering as clu


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
    data = ["Hello, my name is Erik.", "What is that song called?",
               "Tell me the name of that song.", "What year was that song made?"]
    
    model_id = 'sentence-transformers/all-mpnet-base-v2'
    # model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer, model = embed.initialize_embedding_model(model_id)

    pipeline = make_pipeline(
                     lambda data: embed.batch_embeddings(data, tokenizer, model),
                     sim.batch_similarity_scores,
                     clu.knn
                    )
    
    output = pipeline(data)
    
    print(f"Output:\n{output}")
    


