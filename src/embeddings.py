import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, logging
from tqdm import tqdm
from typing import Any, List, Tuple
import gensim.downloader as api
from gensim.models import Word2Vec
import numpy as np

logging.set_verbosity_error()

# -----------------
# Initializers
# -----------------

def initialize_sentence_transformer_model(model_id: str) -> Tuple[Any, Any, int]:
    print(f"Initializing {model_id} Model")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, return_dict=True)
    max_length = tokenizer.model_max_length
    return tokenizer, model, max_length

def initialize_nomic_model() -> Tuple[Any, Any, int]:
    print("Initializing Nomic Model")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)
    model.eval()
    max_length = tokenizer.model_max_length
    return tokenizer, model, max_length

def initialize_google_bert_model() -> Tuple[Any, Any, int]:
    print("Initializing Google BERT Model")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    max_length = tokenizer.model_max_length
    return tokenizer, model, max_length

def initialize_specter_base_model() -> Tuple[Any, Any, int]:
    print("Initializing AllenAI Specter Model")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    model = AutoModel.from_pretrained("allenai/specter")
    model.eval()
    # max_length = tokenizer.model_max_length # returns 1000000000000000019884624838656 lmao
    max_length = 512
    return tokenizer, model, max_length

# def initialize_llama_model() -> Tuple[Any, Any, int]:
#     print("Initializing Llama Model")
#     tokenizer = AutoTokenizer.from_pretrained("deerslab/llama-7b-embeddings", trust_remote_code=True)
#     model = AutoModel.from_pretrained("deerslab/llama-7b-embeddings", trust_remote_code=True)
#     model.eval()
#     max_length = tokenizer.model_max_length
#     return tokenizer, model, max_length 

def initialize_word2vec_model() -> Word2Vec:
    print("Initializing Word2Vec Model")
    model = api.load("word2vec-google-news-300")
    return model # pyright: ignore

# -----------------
# Mean Pooling Functions
# -----------------

def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Perform mean pooling on the token embeddings for sentence transformers."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# -----------------
# Utility Functions
# -----------------

def split_into_chunks(text: str, tokenizer, max_length: int) -> List[str]:
    """Split text into chunks that fit within the model's max context length."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - 2):
        chunk = tokens[i:i + max_length - 2]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
        if len(chunk_tokens) <= max_length:
            chunks.append(chunk_text)
        else:
            sub_chunks = [chunk[j:j + max_length - 2] for j in range(0, len(chunk), max_length - 2)]
            for sub_chunk in sub_chunks:
                sub_chunk_text = tokenizer.decode(sub_chunk, skip_special_tokens=True)
                chunks.append(sub_chunk_text)
    return chunks

def aggregate_embeddings(embeddings: List[torch.Tensor]) -> torch.Tensor:
    """Aggregate embeddings using mean pooling."""
    stacked_embeddings = torch.stack(embeddings, dim=0)
    aggregated_embedding = torch.mean(stacked_embeddings, dim=0)
    return F.normalize(aggregated_embedding, p=2, dim=0)

def document_to_vector(model, doc):
    """Convert a document into a vector by averaging its word vectors."""
    word_vectors = [model[word] for word in doc if word in model]
    if not word_vectors:  # Handle the case where none of the words are in the model
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

# -----------------
# Embedding Functions
# -----------------

def get_embeddings(input: List[str], tokenizer: type[Any], model: type[AutoModel], max_length: int) -> torch.Tensor:
    """Get embeddings for a list of documents."""
    document_embeddings = []
    for document in input:
        chunks = split_into_chunks(document, tokenizer, max_length)
        chunk_embeddings = []
        for chunk in chunks:
            encoded_input = tokenizer(chunk, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
            with torch.no_grad():
                model_output = model(**encoded_input)
            chunk_embedding = mean_pooling(model_output, encoded_input['attention_mask']).squeeze()
            chunk_embeddings.append(chunk_embedding)
        document_embedding = aggregate_embeddings(chunk_embeddings)
        document_embeddings.append(document_embedding)
    return torch.stack(document_embeddings)

def get_word2vec_embeddings(input: List[List[str]], model: Word2Vec) -> torch.Tensor:
    """Get Word2Vec embeddings for a list of documents."""
    document_embeddings = [document_to_vector(model, doc) for doc in input]
    document_embeddings = np.array(document_embeddings)
    return torch.tensor(document_embeddings)

def batch_embeddings(input: List[str], tokenizer: type[AutoTokenizer], model: type[AutoModel], batch_size: int = 32, save_path: str = "", max_length: int = 512) -> torch.Tensor:
    """Get embeddings for a list of documents in batches."""
    embeddings = []
    pbar = tqdm(range(0, len(input), batch_size))
    for i in pbar:
        batch = input[i: i + batch_size]
        batch_embeddings = get_embeddings(batch, tokenizer, model, max_length)
        pbar.set_description(f"Processing batch: {batch[0][:20]}...")
        embeddings.append(batch_embeddings)
        if save_path:
            torch.save(torch.cat(embeddings, dim=0), save_path)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

def batch_word2vec_embeddings(input: List[List[str]], model: Word2Vec, batch_size: int = 32, save_path: str = "") -> torch.Tensor:
    """Get Word2Vec embeddings for a list of documents in batches."""
    embeddings = []
    pbar = tqdm(range(0, len(input), batch_size))
    for i in pbar:
        batch = input[i: i + batch_size]
        batch_embeddings = get_word2vec_embeddings(batch, model)
        pbar.set_description(f"Processing batch: {' '.join(batch[0][:3])}...")
        embeddings.append(batch_embeddings)
        if save_path:
            torch.save(torch.cat(embeddings, dim=0), save_path)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

# -----------------
# Wrapper Function
# -----------------

def get_model_and_pooling_func(model_name: str) -> Tuple[Any, Any, int]:
    """Return the appropriate tokenizer, model, mean pooling function, and max length based on the model name."""
    if model_name == "nomic":
        tokenizer, model, max_length = initialize_nomic_model()
    elif model_name == "minilm":
        model_id = 'sentence-transformers/all-MiniLM-L6-v2'
        tokenizer, model, max_length = initialize_sentence_transformer_model(model_id)
    elif model_name == "mpnet":
        model_id = 'sentence-transformers/all-mpnet-base-v2'
        tokenizer, model, max_length = initialize_sentence_transformer_model(model_id) 
    elif model_name == "bert":
        tokenizer, model, max_length = initialize_google_bert_model()
    elif model_name == "specter":
        tokenizer, model, max_length = initialize_specter_base_model()
    # elif model_name == "llama":
    #     tokenizer, model, max_length = initialize_llama_model()
    elif model_name == "word2vec":
        model = initialize_word2vec_model()
        return None, model, None  # pyright: ignore Word2Vec doesn't use tokenizer and max_length
    else:
        raise ValueError(f"Model name {model_name} is not supported.")
    
    return tokenizer, model, max_length

def process_embeddings(input: List[Any], model_name: str, batch_size: int = 32, save_path: str = "") -> torch.Tensor:
    """Process embeddings based on the specified model name."""
    tokenizer, model, max_length = get_model_and_pooling_func(model_name)
    if model_name == "word2vec":
        embeddings = batch_word2vec_embeddings(input, model, batch_size, save_path)
    else:
        embeddings = batch_embeddings(input, tokenizer, model, batch_size, save_path, max_length)
    return embeddings

if __name__ == "__main__":
    strings = ["Hello, my name is Erik.", "What is that song called?", "Tell me the name of that song.", "What year was that song made?"]
    tokenized_strings = [string.lower().split() for string in strings]
    
    nomic_embeddings = process_embeddings(strings, "nomic")
    print("Nomic Model Embeddings:")
    print(nomic_embeddings.size())
    print(nomic_embeddings)
    
    st_embeddings = process_embeddings(strings,  "minilm")
    print("Sentence Transformer (MiniLM) Embeddings:")
    print(st_embeddings.size())
    print(st_embeddings)

    st_embeddings = process_embeddings(strings,  "mpnet")
    print("Sentence Transformer (MPNet) Embeddings:")
    print(st_embeddings.size())
    print(st_embeddings)

    bert_embeddings = process_embeddings(strings, "bert")
    print("Google BERT Model Embeddings:")
    print(bert_embeddings.size())
    print(bert_embeddings)

    specter_embeddings = process_embeddings(strings, "specter")
    print("AllenAI Specter Model Embeddings:")
    print(specter_embeddings.size())
    print(specter_embeddings)

    word2vec_embeddings = process_embeddings(tokenized_strings, "word2vec")
    print("Word2Vec Model Embeddings:")
    print(word2vec_embeddings.size())
    print(word2vec_embeddings)
