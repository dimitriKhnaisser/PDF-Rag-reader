from app.services.pdf_loader import load_pdf_text
from app.services.chunker import chunk_text

text = load_pdf_text("data/pdfs/sample.pdf")

chunks = chunk_text(text)

print("Total chunks:", len(chunks))
print("\nFirst chunk:\n")
print(chunks[0])