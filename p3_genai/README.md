# Retrieval-Augmented Generation (RAG) System — CPU Only

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system that answers user questions using a custom document corpus instead of relying purely on a language model’s parametric memory.

The system performs:
- Semantic retrieval using vector embeddings
- Context-aware answer generation using a local language model
- End-to-end inference via a FastAPI service

All components run **locally on CPU**, using open-source models.

---

## Problem Statement
Large Language Models can hallucinate when answering questions without grounding.  
This project solves that by retrieving relevant document context before generating answers.

---

## Architecture
↓
Embedding Model
↓
FAISS Vector Search
↓
Top-k Relevant Chunks
↓
Prompt Construction
↓
Local LLM
↓
Answer




---

## Components

### 1. Document Ingestion
- Plain text documents are chunked with overlap
- Chunks are embedded using Sentence Transformers
- Embeddings are indexed using FAISS for fast similarity search

### 2. Retrieval
- User queries are embedded
- FAISS retrieves top-k semantically similar chunks
- Retrieved context is injected into the generation prompt

### 3. Generation
- A local causal language model (`distilgpt2`) generates answers
- Generation is constrained using `max_new_tokens` for stability

### 4. API Layer
- FastAPI exposes a `/query` endpoint
- JSON input → grounded natural language answer

---

## Example Query
```json
{
  "question": "What is Retrieval-Augmented Generation?"
}


Technologies Used

Python

FastAPI

FAISS

Sentence Transformers

Hugging Face Transformers

Pydantic

Uvicorn

Key Learnings

Designing RAG pipelines with clear separation of concerns

Handling prompt-length constraints in generation

Matching model architecture to generation tasks

Debugging real-world GenAI deployment issues

Building CPU-friendly GenAI systems





p3_genai_rag/
├── data/
│   └── documents.txt        # Input knowledge base
├── ingest.py                # Chunking + embedding
├── vector_store.py          # FAISS index logic
├── rag_pipeline.py          # Retrieval + generation
├── llm.py                   # Local LLM wrapper
├── config.py                # Central configuration
├── app.py                   # CLI / simple interface
└── README.md

