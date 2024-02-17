import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    """
    This function takes the model output and attention mask as input and performs mean pooling on the token embeddings.
    Args:
        model_output (tuple): The output of the model, which includes the token embeddings.
        attention_mask (torch.Tensor): The attention mask indicating which tokens are valid.
    Returns:
        torch.Tensor: The mean-pooled token embeddings.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def initialize_embedding_model(model_id):
    print("Initalizing Tokenizer and Model")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id)
    return tokenizer, model


def get_embeddings(input: list[str], tokenizer, model):
    """
    This function takes an input string and returns the embedding of the input using the initialized model.
    If the model is not initialized, it will be initialized before generating the embedding.
    Args:
        input List[str]: Input strings to be embedded.
    Returns:
        torch.Tensor: The embedding of the input string.
    """

    encoded_input = tokenizer(input, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def batch_embeddings(input: list[str], tokenizer, model, batch_size=32, save_path=None):
    """
    This function takes a list of strings and a batch size as input, and returns the embeddings of the strings.
    The embeddings are calculated in batches to optimize memory usage.
    If a save path is provided, the embeddings are saved to the specified path after each batch. Use <name>.pt files.
    Args:
        strings (list[str]): The strings to be embedded.
        batch_size (int, optional): The size of the batches.
        save_path (str, optional): The path where the embeddings should be saved. If None, the embeddings are not saved.
    Returns:
        torch.Tensor: The embeddings of the input strings.
    """
    embeddings = []
    for i in tqdm(range(0, len(input), batch_size)):
        batch = input[i: i+batch_size]
        batch_embeddings = get_embeddings(batch)

        embeddings.append(batch_embeddings)
        if save_path:
            torch.save(torch.cat(embeddings, dim=0), save_path)
    
    embeddings = torch.cat(embeddings, dim=0)

    return embeddings


if __name__ == "__main__":
    pass