import ast
import pandas as pd
import pickle

import src.embeddings as emb
import src.similarity as sim
import src.metrics as met

def load_data(filepath, n=None):
    assert filepath[-4:] == ".csv", "Must be a .csv file"
    data = pd.read_csv(filepath)
    if n:
        data = data.head(n)

    attrs = {
        "titles": data["title"].tolist(),
        "text": data["text"].tolist(),
        "tags": data["tags"].apply(ast.literal_eval).tolist(),
        "ids": data.index.tolist()
    }

    if "simplified_tags" in data.columns:
        attrs["simplified_tags"] = data["simplified_tags"].apply(ast.literal_eval).tolist()

    return attrs


def save_to_pickle(object, filename):
    pickle.dump(object, open(filename, 'wb'))


def load_from_pickle(filename):
    return pickle.load(open(filename, 'rb'))


def get_all_embeddings(data_name, n=10000):
    data = load_data(f"data/{data_name}.csv", n)
    text = data["text"]

    model_name = "minilm"
    print(model_name)
    minilm_embeddings = emb.process_embeddings(text, model_name)
    save_to_pickle(minilm_embeddings, f"embeddings/{data_name}_{model_name}_n{n}.pickle")

    model_name = "mpnet"
    print(model_name)
    mpnet_embeddings = emb.process_embeddings(text, model_name)
    save_to_pickle(mpnet_embeddings, f"embeddings/{data_name}_{model_name}_n{n}.pickle")

    model_name = "nomic"
    print(model_name)
    nomic_embeddings = emb.process_embeddings(text, model_name)
    save_to_pickle(nomic_embeddings, f"embeddings/{data_name}_{model_name}_n{n}.pickle")

    model_name = "bert"
    print(model_name)
    bert_embeddings = emb.process_embeddings(text, model_name)
    save_to_pickle(bert_embeddings, f"embeddings/{data_name}_{model_name}_n{n}.pickle")

    model_name = "specter"
    print(model_name)
    specter_embeddings = emb.process_embeddings(text, model_name)
    save_to_pickle(specter_embeddings, f"embeddings/{data_name}_{model_name}_n{n}.pickle")

    model_name = "word2vec"
    print(model_name)
    word2vec_embeddings = emb.process_embeddings(text, model_name)
    save_to_pickle(word2vec_embeddings, f"embeddings/{data_name}_{model_name}_n{n}.pickle")


if __name__ == "__main__":
    get_all_embeddings("interview_prep")




# data_name = "medium_1k_tags_simplified"
# data = load_data(f"data/{data_name}.csv", 10000)
# text = data["text"]
# tags = data["simplified_tags"]

# # model_name = "minilm"
# # minilm_embeddings = emb.process_embeddings(text, model_name)
# # save_to_pickle(minilm_embeddings, "embeddings/medium1k_minilm_n10000.pickle")

# # model_name = "mpnet"
# # mpnet_embeddings = emb.process_embeddings(text, model_name)
# # save_to_pickle(mpnet_embeddings, "embeddings/medium1k_mpnet_n10000.pickle")

# # model_name = "nomic"
# # nomic_embeddings = emb.process_embeddings(text, model_name)
# # save_to_pickle(minilm_embeddings, "embeddings/medium1k_nomic_n10000.pickle")

# # model_name = "bert"
# # bert_embeddings = emb.process_embeddings(text, model_name)
# # save_to_pickle(minilm_embeddings, "embeddings/medium1k_bert_n10000.pickle")

# # model_name = "specter"
# # bert_embeddings = emb.process_embeddings(text, model_name)
# # save_to_pickle(minilm_embeddings, "embeddings/medium1k_specter_n10000.pickle")

# model_name = "llama"
# llama_embeddings = emb.process_embeddings(text, model_name)
# save_to_pickle(llama_embeddings, "embeddings/medium1k_llama_n10000.pickle")

# model_name = "word2vec"
# word2vec_embeddings = emb.process_embeddings(text, model_name)
# save_to_pickle(word2vec_embeddings, "embeddings/medium1k_word2vec_n10000.pickle")

# # ------------------------------------------------------------------

# data_name = "arXiv0_tags"
# data = load_data(f"data/{data_name}.csv", 10000)
# text = data["text"]
# tags = data["simplified_tags"]

# model_name = "minilm"
# minilm_embeddings = emb.process_embeddings(text, model_name)
# save_to_pickle(minilm_embeddings, "embeddings/arXiv0_minilm_n10000.pickle")

# model_name = "mpnet"
# mpnet_embeddings = emb.process_embeddings(text, model_name)
# save_to_pickle(mpnet_embeddings, "embeddings/arXiv0_mpnet_n10000.pickle")

# model_name = "nomic"
# nomic_embeddings = emb.process_embeddings(text, model_name)
# save_to_pickle(nomic_embeddings, "embeddings/arXiv0_nomic_n10000.pickle")

# model_name = "bert"
# bert_embeddings = emb.process_embeddings(text, model_name)
# save_to_pickle(bert_embeddings, "embeddings/arXiv0_bert_n10000.pickle")

# model_name = "specter"
# specter_embeddings = emb.process_embeddings(text, model_name)
# save_to_pickle(specter_embeddings, "embeddings/arXiv0_specter_n10000.pickle")

# model_name = "llama"
# llama_embeddings = emb.process_embeddings(text, model_name)
# save_to_pickle(llama_embeddings, "embeddings/arXiv0_llama_n10000.pickle")

# model_name = "word2vec"
# word2vec_embeddings = emb.process_embeddings(text, model_name)
# save_to_pickle(word2vec_embeddings, "embeddings/arXiv0_word2vec_n10000.pickle")