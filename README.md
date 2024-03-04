# LociNet

This is my thesis project for the Master of Science in Computer Science at the California Polytechnic State University - San Luis Obispo.

[The Method of Loci](https://en.wikipedia.org/wiki/Method_of_loci)

The method of loci is a strategy for memory enhancement, which uses visualizations of familiar spatial environments in order to enhance the recall of information.

"memory journey, memory palace, journey method, memory spaces, or mind palace technique"

# Code Base

## src modules - what you need to know

`src/embeddings.py` Gets embeddings from corpus using designated model.

`src/evaluation.py` Evaluation metrics to judge resulting graph in different qualities.

`src/graph_generation.py` Generates graph from embedding similarities and clustering information.

`src/similarity.py` Gets similarity between embeddings.

`src/pipeline.py` Compose the modules into a pipeline.

`src/utils.py` General utilities.

# Thoughts

## The Work

Developing a dynamic knowledge graph system to enhance user and machine recall and content exploration.
Utilizing ML techniques, including semantic latent space models, dimensionality reduction, embeddings and geometric graph theory to analyze relationships between user-generated media chunks and external databases to provide content and structural recommendations. Prior work: [LociMaps](https://github.com/loci-maps/mini-map).

## Ideas

General Pipeline

1. Embed data
2. Cluster data (hierarchical) + generate labels (summaries)
3. Connect into directional graph

<img src="assets/clusters.png" alt="clusters" width="250" height="250"/>

Data Simplification - Tags

- [tag]: [article it is summarizing] -> embedding
- k (20?) clusters
- assign label

Metrics and Analysis

- How well a human or AI agent can find a document based on a goal query
- Cooccurence of human vs AI tags: "Group X and Y overlap by this much in B subbranches"
- Compare title + text vs. text vs. title
- Average distance between all nodes
- Average distance between tags
- How long to get from X to Y
