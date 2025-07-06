import os
import json
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR

# --- Configuration ---
CHUNK_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json')
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Agent State (in-memory) ---
mcp = FastMCP("retrieval-agent")
model: Optional[SentenceTransformer] = None
vector_store: Optional[faiss.Index] = None
text_chunks: List[str] = []

def initialize_model():
    global model
    if model is None:
        print("[RetrievalAgent] Loading sentence transformer model...", flush=True)
        model = SentenceTransformer(MODEL_NAME)
        print("[RetrievalAgent] Model loaded.", flush=True)

@mcp.tool()
def build_index() -> Dict[str, Any]:
    """
    Reads text chunks from the data store, generates embeddings,
    and builds a FAISS vector index in memory.
    """
    global vector_store, text_chunks, model
    
    print("[RetrievalAgent] Received request to build index.", flush=True)
    initialize_model()

    if not os.path.exists(CHUNK_DB_PATH):
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Chunk data file not found. Ingestion must run first."))

    with open(CHUNK_DB_PATH, 'r', encoding='utf-8') as f:
        text_chunks = json.load(f)

    if not text_chunks:
        return {"status": "warning", "message": "No text chunks found to index."}

    print(f"[RetrievalAgent] Encoding {len(text_chunks)} chunks...", flush=True)
    embeddings = model.encode(text_chunks, convert_to_tensor=False)
    
    dimension = embeddings.shape[1]
    vector_store = faiss.IndexFlatL2(dimension)
    vector_store.add(np.array(embeddings, dtype=np.float32))
    
    print(f"[RetrievalAgent] FAISS index built successfully with {vector_store.ntotal} vectors.", flush=True)
    return {"status": "success", "vectors_indexed": vector_store.ntotal}

@mcp.tool()
def retrieve(query: str, top_k: int = 5) -> List[str]:
    """
    Finds the most relevant text chunks for a given query.

    Args:
        query: The user's question.
        top_k: The number of chunks to retrieve.
    """
    global vector_store, model
    print(f"[RetrievalAgent] Received query: '{query}'", flush=True)
    initialize_model()
    
    if vector_store is None or not text_chunks:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Vector index not built. Call build_index first."))

    query_embedding = model.encode([query])
    distances, indices = vector_store.search(np.array(query_embedding, dtype=np.float32), top_k)
    
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    print(f"[RetrievalAgent] Retrieved {len(retrieved_chunks)} chunks.", flush=True)
    return retrieved_chunks

if __name__ == "__main__":
    print("[RetrievalAgent] Starting up...", flush=True)
    mcp.run(transport='stdio')