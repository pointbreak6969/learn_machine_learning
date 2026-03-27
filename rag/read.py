import sys
from pypdf import PdfReader
#https://developers.llamaindex.ai/python/framework/module_guides/loading/simpledirectoryreader/
def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n".join(parts)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/DSAP_1.pdf"
    text = pdf_to_text(path)
    print(f"Extracted {len(text)} characters from {path}")
    print(text[:500]) # for previewing the extracted text