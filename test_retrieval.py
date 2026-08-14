from app.services.retriever import retrieve_chunks


question = input("Ask a question: ")

chunks = retrieve_chunks(question, k=10)

print("\nRetrieved chunks:\n")

for i, chunk in enumerate(chunks):

    print(f"\n--- Chunk {i + 1} ---")
    print(f"Source: {chunk['source']}")
    print(chunk["text"])