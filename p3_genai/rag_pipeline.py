"""
Rag_pipeline.py

implements RAG:
-Embeds user query
-Retrieves relevant document chunks
-builds prompt
-generate answer


"""

from sentence_transformers import SentenceTransformer
import numpy as np

from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH
from vector_store import load_faiss_index
from llm import LocalLLM

class RAGPipeline:
    """
    End-to-End RAG pipeline
    """

    def __init__(self):
        """
        Load embedding mode, FAISS index , and LLM:
        """
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index, self.documents = load_faiss_index(FAISS_INDEX_PATH)
        self.llm  = LocalLLM()

    def retrieve(self, query:str, top_k:int =3):
        """
        Retrieve top-k relevant document chunks.
        Args:
            query: User query
            top_k : Number of Chunks to retrieve
        Returns :
            List[str]: Relevant document chunks
        """

        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), top_k

        )

        return [self.documents[i] for i in indices[0]]
    
    def generate_answer(self,query:str)-> str:
        """
        Generate answer using retrieved context.

        Args:
        query: User query

        Returns:
        str: Generated response

        """

        retrieved_docs = self.retrieve(query)

        #Construct prompt with retrieved docs
        context = "\n".join(retrieved_docs)

        prompt= (

            "Use the following context to answer the question. \n\n"
            f"Context: \n{context}\n\n"
            f"Question: {query}\n Answer:"
        )

        return self.llm.generate(prompt)