import embeddings as em
import seaborn as sns
from nltk.tokenize import sent_tokenize










def split_into_sections(text):
    sentences = sent_tokenize(text)
    embeddings = em.get_embeddings(sentences)
    similarities = em.similarity_rankings(embeddings, embeddings)[:, 1:]
    sns.heatmap(similarities, annot=True)



if __name__ == "__main__":
    text = "Then I awoke, like a man drained of blood, who wanders alone in a waste. Thank you once again for listening to the Fall of Civilization's podcast. I'd like to thank my voice actors for this episode. Re Brignell, jake Barrett Mills, shem Jacobs, Nick Bradley and Emily Johnson. I love to hear your thoughts and responses on Twitter, so please come and tell me what you thought. You can follow me at Paul M. This podcast can only keep going with the support of our generous subscribers on Patreon. You keep me running, you help me cover my costs, and you help keep the podcast ad free. ... If you enjoyed this podcast, please consider heading on to Patreon.com."
    split_into_sections(text)