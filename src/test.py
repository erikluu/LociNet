import pinecone_utils as pi
from config.config import load_config


config = load_config("config/config.yaml")

pi.init_pineconeDB(config['pinecone'])

stats = pi.get_index_stats()


