"""
ingest.py

Pipeline for:
-Reading documents
-chunking text
-creating embeddings
-Building and saving FAISS index

"""

from sentence_transformers import SentenceTransformer
import faiss

from config import (
    DOCUMENT_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH
)

from vector_store import save_faiss_index


def load_documents(path):
    """
    Loads raw text documents from file
    :param path: Description

    Returns:
    str: RAW documents text

    """
    with open(path, "r", encoding= "utf-8") as f:
        return f.read()
    
def chunk_text(text, chunk_size, overlap):
    """
    Splits text into overlapping texts 
    
    :param text: Full document text
    :param chunk_size: Characters per chunk
    :param overlap: Overlap between chunks

    Returns:
        List[str]: List of text chunks
    """
    chunks =[]
    start = 0

    while start< len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end-overlap

    
    return chunks


def ingest():

    #load raw documents
    text = load_documents(DOCUMENT_PATH)
    print(f"Loaded document with {len(text)} characters")

    #Chunk documents
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} text chunks")

    # Load embedding model (CPU)
    print("Loading embedding model...")
    model =  SentenceTransformer(EMBEDDING_MODEL_NAME)

    #Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar = True)


    #Create FAISS index

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    #Save index and chunks

    save_faiss_index(index, chunks, FAISS_INDEX_PATH)

    print(f"ingestion complete. Indeced {len(chunks)} chunks.")



if __name__ == "__main__":
    ingest()


