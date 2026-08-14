import faiss

index = faiss.read_index("data/vectorstore/index.faiss")

print("Number of vectors:", index.ntotal)
print("Vector dimension:", index.d)