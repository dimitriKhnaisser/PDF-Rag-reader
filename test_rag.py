from app.services.pdf_loader import load_pdf_text
from app.services.chunker import chunk_text
from app.services.embeddings import embed_texts
from app.services.vector_store import VectorStore
from app.services.llm import generate_answer

# 1. Load PDF
text = load_pdf_text("data/pdfs/sample.pdf")

# 2. Chunk
chunks = chunk_text(text)

# 3. Embed + store
vectors = embed_texts(chunks)
store = VectorStore(dimension=len(vectors[0]))
store.add(vectors, chunks)

# 4. Query
# question = "What is Dimitri studying?"
question = "Give me 7 of the top skills Dimitri has?"

query_vector = embed_texts([question])[0]

retrieved_chunks = store.search(query_vector, k=3)

# 5. LLM answer
answer = generate_answer(question, retrieved_chunks)

print("\n--- ANSWER ---\n")
print(answer)