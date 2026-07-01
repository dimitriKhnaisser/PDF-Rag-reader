from app.services.pdf_loader import load_pdf_text
from app.services.chunker import chunk_text
from app.services.embeddings import embed_texts
from app.services.vector_store import VectorStore

text = load_pdf_text("data/pdfs/sample.pdf")
chunks = chunk_text(text)

vectors = embed_texts(chunks)

store = VectorStore(dimension=len(vectors[0]))
store.add(vectors, chunks)

query = "What is Dimitri studying?"
query_vector = embed_texts([query])[0]

results = store.search(query_vector)

print("\n--- RESULTS ---\n")
for r in results:
    print(r)
    print("------")


# we add the chunks of the file we have, store.add
# then do store.search(query), query is the question where we want to get an answer from the pdf
# so by doing this the results will be all the chunks vectors that are close to the query one