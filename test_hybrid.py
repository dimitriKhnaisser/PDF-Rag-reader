from app.services.hybrid_retriever import retrieve_hybrid


question = input("Ask a question: ")

chunks = retrieve_hybrid(question, k=5)

print("\nRetrieved chunks:\n")

for i, chunk in enumerate(chunks):

    print(f"\n--- Chunk {i + 1} ---")
    print(f"Source: {chunk['source']}")
    print(chunk["text"])