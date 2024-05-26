import numpy as np
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt

import embeddings as em

def rev_sigmoid(x):
    return (1 / (1 + np.exp(0.5*x)))


def activate_similarities(similarities):
    """
    Activates the similarities by applying a sigmoid function to the diagonal elements of the similarity matrix.

    Args:
        similarities (numpy.ndarray): The similarity matrix.

    Returns:
        numpy.ndarray: The activated similarities.

    """
    x = np.linspace(-10, 10, similarities.shape[0])
    y = np.vectorize(rev_sigmoid)

    activation_weights = y(x)

    diagonals = [similarities.diagonal(each) for each in range(0, similarities.shape[0])]
    diagonals = [np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals]
    diagonals = np.stack(diagonals)
    diagonals = diagonals * activation_weights.reshape(-1, 1)

    activated_similarities = np.sum(diagonals, axis=0)

    return activated_similarities


def split_into_sections(text):
    """
    Splits the given text into sections based on the similarity of sentences.

    Args:
        text (str): The input text.

    Returns:
        list: A list of paragraphs, where each paragraph is a list of sentence indices.

    """
    sentences = sent_tokenize(text)
    embeddings = em.get_embeddings(sentences)
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

    activated_similarities = activate_similarities(similarity_matrix)
    minmimas = argrelextrema(activated_similarities, np.less, order=2)

    split_points = [s for s in minmimas[0]]

    paragraphs = []
    paragraph = []
    for i, p in enumerate(split_points):
        if i in split_points:
            paragraphs.append(paragraph)
            paragraph = [p]
        else:
            paragraph.append(p)
        
    return paragraphs


if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    text = "Then I awoke, like a man drained of blood, who wanders alone in a waste. Thank you once again for listening to the Fall of Civilization's podcast. I'd like to thank my voice actors for this episode. Re Brignell, jake Barrett Mills, shem Jacobs, Nick Bradley and Emily Johnson. I love to hear your thoughts and responses on Twitter, so please come and tell me what you thought. You can follow me at Paul M. This podcast can only keep going with the support of our generous subscribers on Patreon. You keep me running, you help me cover my costs, and you help keep the podcast ad free. ... If you enjoyed this podcast, please consider heading on to Patreon.com."
    print(split_into_sections(text))