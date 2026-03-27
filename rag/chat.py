import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

DATA_DIR = "data"
EMBED_MODEL= "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "Qwen/Qwen2-7B-Instruct"

MODEL_CACHE = os.path.abspath(".cache/models")
os.makedirs(".cache/huggingface/hub", exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.abspath(".cache/huggingface"))

# --- 2. DATA PROCESSING ---
def load_all_pdf_text() -> tuple[list[str], list[str]]:
    """Reads PDFs and cuts them into manageable chunks."""
    chunks_out = []
    sources_out = []
    
    if not os.path.exists(DATA_DIR):
        return [], []

    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            if not name.lower().endswith(".pdf"):
                continue
            
            path = os.path.join(root, name)
            try:
                reader = PdfReader(path)
                # Combine all page text into one string
                full_text = ""
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        full_text += content + " "
                
                # Split text into words for "Chunking"
                words = full_text.split()
                chunk_size, overlap = 300, 50
                
                i = 0
                while i < len(words):
                    # Create a snippet of 300 words
                    snippet = " ".join(words[i : i + chunk_size])
                    if snippet.strip():
                        chunks_out.append(snippet)
                        # Keep relative path so same filename in different folders is not ambiguous
                        sources_out.append(os.path.relpath(path, DATA_DIR))
                    # Slide the window forward, keeping 50 words from the previous chunk
                    i += chunk_size - overlap
                    
            except Exception as e:
                print(f"Error reading {name}: {e}")
                
    return chunks_out, sources_out

# --- 3. THE SEARCH ENGINE (INDEXING) ---
def build_index(chunks: list[str], embedder: SentenceTransformer):
    """Converts text chunks into a searchable mathematical index."""
    # Step 1: Turn text into numbers (Embeddings)
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    
    # Step 2: Initialize FAISS (L2 = Euclidean distance / 'straight line' distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Step 3: Add the numbers to the index
    index.add(embeddings.astype("float32"))
    return index

# --- 4. RETRIEVAL LOGIC ---
def retrieve(query: str, embedder, index, chunks, sources, k: int = 3):
    """Finds the 'k' most relevant pieces of text for a question."""
    if not chunks:
        return []
    
    # Turn the user's question into numbers
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    
    # Search the index for the 'k' closest matches
    k_eff = min(k, len(chunks))
    distances, indices = index.search(query_vec, k_eff)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue  # Skip invalid matches
        results.append({
            "text": chunks[idx],
            "source": sources[idx],
            "score": float(np.exp(-dist)) # Convert distance to a 0-1 similarity score
        })
    return results

# --- 5. THE AI BRAIN (GENERATION) ---
def answer_question(question: str, context_hits, generator):
    """Sends the context and question to the LLM for a final answer."""
    if not context_hits:
        context_str = "No relevant information found in documents."
    else:
        # Format the snippets into a single "Context" block
        context_parts = [f"Source: {h['source']}\nContent: {h['text']}" for h in context_hits]
        context_str = "\n---\n".join(context_parts)

    # The "System Prompt" tells the AI how to behave
    prompt = (
        "You are a helpful assistant. Use the provided context to answer the question. "
        "If the answer isn't in the context, say you don't know. Be concise.\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    response = generator(
        prompt,
        max_new_tokens=256, # specifies the maximum length of the generated answer
        temperature=0.3, # Low temperature = more factual, less creative
        do_sample=True, # Whether to use sampling; if False, uses greedy decoding
        truncation=True # Truncate the prompt if it's too long for the model's context window
    )
    
    # Clean up the output to show only the AI's new text
    full_text = response[0]["generated_text"]
    return full_text[len(prompt):].strip()

# --- 6. MAIN PROGRAM ---
def main():
    print("Step 1: Loading and Chunking PDFs...")
    chunks, sources = load_all_pdf_text()
    
    if not chunks:
        print(f"Error: No PDF text found. Please put PDFs in the '{DATA_DIR}' folder.")
        return

    print(f"Step 2: Creating embeddings for {len(chunks)} chunks from {len(set(sources))} PDF file(s)...")
    embedder = SentenceTransformer(EMBED_MODEL)
    index = build_index(chunks, embedder)

    print("Step 3: Loading AI Generation Model (this may take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, cache_dir=MODEL_CACHE)
    
    # Fix for certain models that don't have a clear "stop" or "padding" token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, 
        cache_dir=MODEL_CACHE, 
        device_map="auto", # Automatically uses GPU if available
        torch_dtype="auto"
    )
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print("\n--- System Ready! ---")
    while True:
        user_input = input("\nAsk a question (or type 'exit'): ").strip()
        if user_input.lower() in ["exit", "quit", ""]:
            break
            
        # 1. Retrieve
        hits = retrieve(user_input, embedder, index, chunks, sources)
        
        # 2. Generate
        answer = answer_question(user_input, hits, generator)
        
        print(f"\n[Sources used: {', '.join(set(h['source'] for h in hits))}]")
        print(f"AI: {answer}")

if __name__ == "__main__":
    main()