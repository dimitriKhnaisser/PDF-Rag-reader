import pickle

CHUNKS_PATH = "data/vectorstore/chunks.pkl"

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(f"Source: {chunk['source']}")
    print(chunk["text"])