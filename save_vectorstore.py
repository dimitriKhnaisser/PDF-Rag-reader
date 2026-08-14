import os
import pickle
import faiss

from app.services.pdf_loader import load_pdf_text
from app.services.chunker import chunk_text
from app.services.embeddings import embed_texts


PDF_FOLDER = "data/pdfs"
VECTORSTORE_FOLDER = "data/vectorstore"

all_chunks = []


# Process every PDF
for filename in os.listdir(PDF_FOLDER):

    if filename.lower().endswith(".pdf"):

        pdf_path = os.path.join(PDF_FOLDER, filename)

        print(f"Processing: {filename}")

        # Load PDF
        text = load_pdf_text(pdf_path)

        # Split into chunks
        chunks = chunk_text(text)

        # Save each chunk together with its source
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": filename
            })


# Get only the text for embedding
texts = [item["text"] for item in all_chunks]

# Create embeddings
vectors = embed_texts(texts)


# Create FAISS index
dimension = len(vectors[0])

index = faiss.IndexFlatL2(dimension)

index.add(vectors)


# Create vectorstore folder
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)


# Save FAISS
faiss.write_index(
    index,
    os.path.join(VECTORSTORE_FOLDER, "index.faiss")
)


# Save chunks + metadata
with open(
    os.path.join(VECTORSTORE_FOLDER, "chunks.pkl"),
    "wb"
) as f:
    pickle.dump(all_chunks, f)


print()
print(f"Total chunks saved: {len(all_chunks)}")
print("FAISS index saved.")
print("Chunks and metadata saved.")