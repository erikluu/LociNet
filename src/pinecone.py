import os
import pinecone
import time
import tqdm
import yaml

class BatchSchema:
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
    def get

def upsert_files(files):
    for file_path in files:




