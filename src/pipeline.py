import embeddings as embed
import similarity as sim
import clustering as clus

# MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'


def compose(*functions):
    def composed_function(data):
        result = data
        for function in reversed(functions):
            result = function(result)
        return result
    return composed_function


def pipeline(data, *functions):
    return compose(*functions)(data)


if __name__ == "__main__":
    data = ["Hello, my name is Erik.", "What is that song called?",
               "Tell me the name of that song.", "What year was that song made?"]
    
    model_id = 'sentence-transformers/all-mpnet-base-v2'
    tokenizer, model = embed.initialize_embedding_model(model_id)

    output = pipeline(data,
                     lambda data: embed.batch_embeddings(data, tokenizer, model),
                     sim.batch_similarity_rankings
                     )
    
    print(output)
    


