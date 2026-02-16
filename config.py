BASE_PATH = "F:/offline_wiki_gpt"  # your external drive

RAW_DATA_PATH = f"{BASE_PATH}/data/raw"
PROCESSED_PATH = f"{BASE_PATH}/data/processed"
INDEX_PATH = f"{BASE_PATH}/data/index/faiss.index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5
