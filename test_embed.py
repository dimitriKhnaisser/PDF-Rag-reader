from app.services.pdf_loader import load_pdf_text
from app.services.chunker import chunk_text
from app.services.embeddings import embed_texts

text = load_pdf_text("data/pdfs/sample.pdf")
chunks = chunk_text(text)

vectors = embed_texts(chunks)

print("Chunks:", len(chunks))
print("Vector shape:", len(vectors), len(vectors[0]))