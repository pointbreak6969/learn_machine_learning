# 02_chunk.py

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

if __name__ == "__main__":
    from pypdf import PdfReader

    def pdf_to_text(pdf_path):
        reader = PdfReader(pdf_path)
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n".join(parts)
    
    text = pdf_to_text("data/DSAP_1.pdf")
    chunks = chunk_words(text, chunk_size=500, overlap=50)
    print(f"Total chunks created: {len(chunks)}")
    if not chunks:
        print("No chunks created.")
    else:
        print(f"First chunk preview: {chunks[12]}")








# 17000 words
# chunk1 = first word to 5000th word
# chunk2 = 4001st word to 9000th word
# chunk3 = 8001st word to 13000th word
# chunk4 = 12001st word to 17000th word