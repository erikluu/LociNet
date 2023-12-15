import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = None
model = None

def similarity_rankings(em0, em1, k=None, threshold=0.0):
    """
    Compute top-k cosine similarity between a set of embeddings and itself.

    Args:
        sections (torch.Tensor): Tensor containing embeddings of sections to cite.
        references (torch.Tensor): Tensor containing possible reference embeddings.
        k (int): Number of top similarities to retrieve for each section.
    Returns:
        dict: A dictionary where keys are indices of sections,
              and values are lists of tuples containing top-k values and indices from the references.
    """
    assert em0.size(-1) == em1.size(-1), f"Dimensions of em0 ({em0.size(-1)}) and em1 ({em1.size(-1)}) do not match."

    cosine_similarity_matrix = F.cosine_similarity(em0.unsqueeze(1), em1.unsqueeze(0), dim=2)
    rankings = torch.argsort(cosine_similarity_matrix, dim=1, descending=True)
    rankings = rankings[:, :k] if k else rankings
    rankings = rankings[cosine_similarity_matrix > threshold]
    
    return rankings


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


def get_embeddings(input: list[str]):
    """
    This function takes an input string and returns the embedding of the input using the initialized model.
    If the model is not initialized, it will be initialized before generating the embedding.
    Args:
        input List[str]: Input strings to be embedded.
    Returns:
        torch.Tensor: The embedding of the input string.
    """
    global tokenizer, model

    if tokenizer is None or model is None:
        print("Initalizing Tokenizer and Model")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        model = AutoModel.from_pretrained(MODEL_ID)

    encoded_input = tokenizer(input, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


if __name__ == "__main__":
    pass




