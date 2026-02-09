"""
config.py

Central configuration for the RAG System
Keeping all constants here makes the project clean and easy to modify

"""
import os

# Base directory of the p3_genai project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to raw documents
DOCUMENT_PATH = os.path.join(BASE_DIR, "data", "documents.txt")

#Chunking Parameters

CHUNK_SIZE = 200            #Number of characters per chunk
CHUNK_OVERLAP = 50          #Overlap to preserve context across chunks

#Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

#Vector Store path

# Vector store path
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")


