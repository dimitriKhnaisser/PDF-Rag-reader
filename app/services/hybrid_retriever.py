import pickle
import faiss

from rank_bm25 import BM25Okapi

from app.services.embeddings import embed_texts


INDEX_PATH = "data/vectorstore/index.faiss"
CHUNKS_PATH = "data/vectorstore/chunks.pkl"


def retrieve_hybrid(question: str, k: int = 5):

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # -----------------------------
    # FAISS
    # -----------------------------

    query_vector = embed_texts([question])

    distances, faiss_indices = index.search(
        query_vector,
        len(chunks)
    )

    faiss_ranking = list(faiss_indices[0])


    # -----------------------------
    # BM25
    # -----------------------------

    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_question = question.lower().split()

    bm25_scores = bm25.get_scores(tokenized_question)

    bm25_ranking = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )


    # -----------------------------
    # Reciprocal Rank Fusion
    # -----------------------------

    rrf_scores = {}

    for rank, index_number in enumerate(faiss_ranking):
        rrf_scores[index_number] = (
            rrf_scores.get(index_number, 0)
            + 1 / (60 + rank + 1)
        )

    for rank, index_number in enumerate(bm25_ranking):
        rrf_scores[index_number] = (
            rrf_scores.get(index_number, 0)
            + 1 / (60 + rank + 1)
        )


    # -----------------------------
    # Final ranking
    # -----------------------------

    ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    top_indices = ranked_indices[:k]

    return [chunks[i] for i in top_indices]