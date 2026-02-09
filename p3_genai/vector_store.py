"""
vector_store.py
Responsible for:
-Creating a FAISS vector index
-Saving the index to disk
-Loading the index for retrieval


Docstring for p3_genai.vector_store
"""
import faiss
import os
import pickle

def save_faiss_index(index, metadata, path):
    """

    Saves FAISS index and metadata to disk.

    :param index: FAISS index object
    :param metadata: List of text chunks
    :param path: Directory to save index files
    """
    os.makedirs(path, exist_ok=True)
    
    #SAVE FAISS INDEX
    faiss.write_index(index, os.path.join(path,"index.faiss"))

    #SAVE metadata( text chunks )
    with open(os.path.join(path, "metadata.pkl"),"wb") as f:
        pickle.dump(metadata,f) 

def load_faiss_index(path):
    """
    Loads FAISS index and metadata from disk
    
    :param path: Directory where index is stored

    returns:
    index, metadata
    """
    index = faiss.read_index(os.path.join(path, "index.faiss"))

    with open(os.path.join(path, "metadata.pkl"),"rb") as f:
        metadata = pickle.load(f)

    return index, metadata