import embeddings as em
import clustering as cl

# MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_ID = 'sentence-transformers/all-mpnet-base-v2'

def compose(*functions):
    def composed_function(data):
        result = data
        for function in reversed(functions):
            result = function(result)
        return result
    return composed_function


def pipeline(data, tokenizer, model, *functions):
    return compose(*functions)(data, tokenizer, model)


if __name__ == "__main__":
    data = []
    tokenizer, model = em.initialize_embedding_model()
    graph = pipeline(data,
                     lambda data: em.batch_embeddings(data, tokenizer, model),
                     em.batch_similarity_rankings_2d)
    


