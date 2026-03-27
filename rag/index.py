# 03_index.py

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2



import os

os.makedirs(".cache/huggingface/hub", exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.abspath(".cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.abspath(".cache/huggingface/hub"))

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n".join(parts)


def chunk_words(text, chunk_size, overlap):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size.")
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        piece = words[i:i + chunk_size]
        chunks.append(' '.join(piece))
        i += chunk_size - overlap
    return chunks


def main():
    pdf_path = "data/DSAP_1.pdf"
    text = pdf_to_text(pdf_path)
    chunks = chunk_words(text, chunk_size=500, overlap=50)
    print(f"Total chunks created: {len(chunks)}")
    if not chunks:
        print("No chunks created.")
        return
    
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=os.path.abspath(".cache/sentence-transformers"),
        )
    
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    print(f"FAISS index created with {index.ntotal} vectors of dimension {dim}.")

    # Example query
    query = "What is digital signal processing?"
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    dist, idx = index.search(query_embedding, k=5)
    print("Top 5 similar chunks:")
    for rank, j in enumerate(idx[0]):
        if j < 0 or j >= len(chunks):
            print(f"Rank {rank + 1}: Invalid index {j}")
            continue
        print(f"Rank {rank + 1}: Chunk index {j}, Distance: {dist[0][rank]:.4f}")
        print(f"Chunk content preview: {chunks[j][:200]}...\n")


if __name__ == "__main__":
    main()