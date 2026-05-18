from __future__ import annotations
# To avoid circular imports when using type hints

import streamlit as st
import importlib.util
from pathlib import Path



def _load_rag_chat_module():
    # Dynamically load the 04_rag_chat.py module
    root = Path(__file__).resolve().parent
    path = root / "04_rag_chat.py"
    spec = importlib.util.spec_from_file_location("rag_capstone", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner="Loading PDFs, Embeddings and Generator...")
def load_pipeline():
    """Load the RAG pipeline components from 04_rag_chat.py and cache them for Streamlit."""
    rag = _load_rag_chat_module()
    chunks, sources = rag.load_all_pdf_text()
    if not chunks:
        return {"ok": False, "error": f"No PDF text found. Please put PDFs in the '{rag.DATA_DIR}' folder."}
    
    embedder = rag.SentenceTransformer(rag.EMBED_MODEL)
    index = rag.build_index(chunks, embedder)

    rag.os.makedirs(rag.MODEL_CACHE, exist_ok=True)
    tokenizer = rag.AutoTokenizer.from_pretrained(rag.GEN_MODEL, cache_dir=rag.MODEL_CACHE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = rag.AutoModelForCausalLM.from_pretrained(
        rag.GEN_MODEL, 
        cache_dir=rag.MODEL_CACHE, 
        device_map="auto",
        torch_dtype="auto"
    )

    generator = rag.pipeline("text-generation", model=model, tokenizer=tokenizer)

    return {
        "ok": True,
        "chunks": chunks,
        "sources": sources,
        "embedder": embedder,
        "index": index,
        "generator": generator,
    }
def main():
    st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📚", layout="centered")
    st.title("📚 RAG PDF Chatbot")
    st.caption("Ask questions about the content of your PDFs!")

    pipe = load_pipeline()
    if not pipe["ok"]:
        st.error(pipe["error"])
        st.stop()
    
    rag = pipe["rag"]
    chunks = pipe["chunks"]
    sources = pipe["sources"]
    embedder = pipe["embedder"]
    index = pipe["index"]
    generator = pipe["generator"]

    with st.sidebar:
        st.header("Settings")
        k = st.slider("Number of Retrieved Chunks (k)", min_value=1, max_value=10, value=3)
        st.divider()
        st.markdown(
            "Same pipeline as in `04_rag_chat.py`, but with Streamlit UI and caching for faster load times. "
            "Make sure to have your PDFs in the `data/` folder and adjust the `GEN_MODEL` and `EMBED_MODEL` in `04_rag_chat.py` if needed."
            "Reload the page after adding PDFs to see them in the chatbot."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
        else:
            st.markdown(f"**AI:** {m['content']}")
    
    prompt = st.chat_input("Ask a question about your PDFs...")
    if not prompt:
        return
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("AI", avatar="🤖"):
        with st.spinner("Generating answer..."):
            hits = rag.retrieve(prompt, embedder, index, chunks, sources, k=k)
            answer = rag.answer_question(prompt, hits, generator)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
                


if __name__ == "__main__":
    main()
