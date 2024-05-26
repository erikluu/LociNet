from dataclasses import dataclass
import embeddings as em
import datetime
import torch
from nltk.tokenize import PunktParagraphTokenizer


paragraph_tokenizer = PunktParagraphTokenizer()

def parse_sections(text):
    paragraphs = paragraph_tokenizer.tokenize(text)
    embeddings = em.get_embeddings(paragraphs)
    return zip(paragraphs, embeddings)



def initialize_document(document):
    id: document['title']
    sections = parse_sections(document['text'])
    created = datetime.datetime.now()
    last_modified = created
    return {
        'id': id,
        'sections': sections,
        'created': created,
        'last_modified': last_modified
    }


if __name__ == "__main__":
    with open("../data/XAI.txt", 'r') as f:
        plain_text = f.read()
    
    doc = initialize_document(
        {
            'title': "XAI.txt",
            'text': plain_text
        })