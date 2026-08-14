from app.services.retriever import retrieve_chunks
from app.services.llm import generate_answer


question = input("Ask a question: ")

# Retrieve relevant chunks from saved FAISS
retrieved_chunks = retrieve_chunks(question, k=10)

# Generate answer using the retrieved chunks
answer = generate_answer(question, retrieved_chunks)

print("\nAnswer:")
print(answer)