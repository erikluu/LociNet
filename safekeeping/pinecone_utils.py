from dotenv import load_dotenv
import os
import pinecone
import time
from tqdm import tqdm

load_dotenv()

index = None

def init_pineconeDB(config):
    """
    Initializes the PineconeDB index with the given parameters.

    Args:
        config (dict): A dictionary containing the configuration parameters for initializing the PineconeDB index.
            - index_name (str): The name of the index.
            - embedding_length (int): The length of the embedding vectors.
            - metric (str): The distance metric to be used.

    Returns:
        Dictionary of index statistics | None
    """
    global index
    index_name = config['index_name']

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=config['embedding_length'],
            metric=config['metric']
        )
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)

    try:
        index = pinecone.Index(index_name)
        return index.describe_index_stats()
    except pinecone.exceptions.IndexNotFoundError:
        return None
    else:
        print(f"PineconeDB initialized. Index name: {index_name}")
        return index.describe_index_stats()


def get_index_stats():
    return index.describe_index_stats()


def upsert_documents(docs):
    """
    Upserts the given files into the Pinecone index.
    Requires (id: str, vector: List[float])

    Args:
        docs (list): A list of files to upsert.
            - each of the form:
                {
                    id: <str>,
                    values: List[float],
                    metadata: {
                        type: "cluster"|"document"|"chunk",
                        time_created: <datetime>,
                        time_modified: <datetime>,
                        hyperdocs: List[ids],
                        hypodocs: List[ids],
                        keywords: List[str]
                    }

                }

    Returns:
        bool: True if the upsert is successful, False otherwise.
    """
    
    if not index:
        raise "Pinecone index not initialized. Run init_pineconeDB()"

    docs_count = len(docs)
    batch_size = 32
    for i in tqdm(range(0, docs_count, batch_size)):
        i_end = min(docs_count, i+batch_size)
        batch = docs[i:i_end]
        index.upsert(vectors=batch)
    
    return True

def query(vector, k=3, _filter={}, include_values=False, include_metadata=True):
    """
    This function performs a query on the Pinecone index.
    Avoid include_metadata True when k ≥ 1000
    
    Args:
        vector (numpy.ndarray): The query vector.
        k (int, default=3): The number of nearest neighbors to retrieve.
        _filter (dict, optional): The filter to apply to the query results. Example:
            {
                "genre": {"$eq": "documentary"},
                "year": 2019
            }
        include_values (bool, optional): Whether to include the values in the query results. Defaults to False.
        include_metadata (bool, optional): Whether to include the metadata in the query results. Defaults to True.
    
    Returns:
        dict: The query response.
    """
    if not index:
        raise "Pinecone index not initialized. Run init_pineconeDB()"

    response = index.query(
        vector=vector,
        top_k=k,
        filter=_filter,
        include_values=include_values,
        include_metadata=include_metadata
    )

    return response

def retrieve_all_clusters():
    response = index.query(
        filter={
            type: "cluster"
        }
    )

    return response
