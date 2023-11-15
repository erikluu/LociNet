from dotenv import load_dotenv
import os
import pinecone
import time
from tqdm import tqdm

load_dotenv()

index = None

def init_pineconeDB(index_name, embedding_length, metric='cosine'):
    """
    Initializes the PineconeDB index with the given parameters.

    Args:
        index_name (str): The name of the index.
        embedding_length (int): The length of the embedding vectors.
        metric (str, optional): The distance metric to use. Dependent on embedding model used. Defaults to 'cosine'. 
    """
    global index

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=embedding_length,
            metric=metric # dependent on model
        )
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)

    try:
        index = pinecone.Index(index_name)
        index.describe_index_stats()
    except pinecone.exceptions.IndexNotFoundError:
        print("Index does not exist.")
    else:
        print(f"PineconeDB initialized. Index name: {index_name}")


def get_index_stats():
    return index.describe_index_stats()


def upsert_files(files):
    """
    Upserts the given files into the Pinecone index.
    Requires (id: str, vector: List[float])

    Args:
        files (list): A list of files to upsert.

    Returns:
        bool: True if the upsert is successful, False otherwise.
    """
    
    if not index:
        raise "Pinecone index not initialized. Run init_pineconeDB()"

    batch_size = 32
    for i in tqdm(range(0, len(files), batch_size)):
        i_end = min(len(files), i+batch_size)
        batch = files[i:i_end]
        index.upsert(vectors=batch)
    
    return True

def vector_query(vector, k=3, _filter={}, include_values=False, include_metadata=True):
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