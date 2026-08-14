import pickle
import faiss

from app.services.embeddings import embed_texts


INDEX_PATH = "data/vectorstore/index.faiss"
CHUNKS_PATH = "data/vectorstore/chunks.pkl"


def retrieve_chunks(question: str, k: int = 5):

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    query_vector = embed_texts([question])

    distances, indices = index.search(query_vector, k)

    retrieved_chunks = []

    for distance, index_number in zip(distances[0], indices[0]):

        print(f"Distance: {distance:.4f} | Chunk index: {index_number}")

        retrieved_chunks.append(chunks[index_number])

    return retrieved_chunks