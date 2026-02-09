"""
app.py

FastAPI application exposing a RAG-based question-answering endpoint

"""

from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import RAGPipeline

#Initialize FASTapi App

app = FastAPI(title="RAG QA system")

#Load RAG Pipeline once at startup

rag_pipeline = RAGPipeline()


class QueryRequest(BaseModel):

    """
    Request schema for /query endpoint/

    """
    question:str


class QueryResponse(BaseModel):

    """
    Request schema for /query endpoint/

    """
    answer:str

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Accepts a user question and returns a generated answer.
    Args:
        request (QueryRequest): User question
    
    Returns:
        QueryResponse: Generated answer
    """
    answer = rag_pipeline.generate_answer(request.question)
    return QueryResponse(answer = answer)