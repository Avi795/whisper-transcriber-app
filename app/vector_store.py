# app/vector_store.py
import faiss
import numpy as np

def create_faiss_index(embeddings):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def search_faiss_index(index, query_embedding, chunks, k=3):
    D, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]

