import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def initialize_embedding_model(model_id: str):
    if not AutoTokenizer.from_pretrained(model_id, use_fast=True).is_fast: # is cached check
        print("Initalizing Tokenizer and Model")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id, return_dict=True)
    return tokenizer, model


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    This function takes the model output and attention mask as input and performs mean pooling on the token embeddings.
    Args:
        model_output (tuple): The output of the model, which includes the token embeddings.
        attention_mask (torch.Tensor): The attention mask indicating which tokens are valid.
    Returns:
        torch.Tensor: The mean-pooled token embeddings.
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeddings(input: list[str], tokenizer: AutoTokenizer, model: AutoModel) -> torch.Tensor:
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


def batch_embeddings(input: list[str], tokenizer: AutoTokenizer, model: AutoModel, batch_size: int = 32, save_path: str = "") -> torch.Tensor:
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
    pbar = tqdm(range(0, len(input), batch_size))
    for i in pbar:
        batch = input[i: i+batch_size]
        batch_embeddings = get_embeddings(batch, tokenizer, model)

        pbar.set_description(f"Processing batch: {batch[0][:20]}...")

        embeddings.append(batch_embeddings)
        if save_path:
            torch.save(torch.cat(embeddings, dim=0), save_path)

    embeddings = torch.cat(embeddings, dim=0)

    return embeddings


if __name__ == "__main__":
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer, model = initialize_embedding_model(model_id)

    strings = ["Hello, my name is Erik.", "What is that song called?",
               "Tell me the name of that song.", "What year was that song made?"]

    embeddings = batch_embeddings(strings, tokenizer, model)
    print(embeddings.size())
    print(embeddings)
