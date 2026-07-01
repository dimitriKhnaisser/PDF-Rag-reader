import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

    def add(self, vectors, chunks):
        vectors = np.array(vectors).astype("float32")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector, k=3):
        query_vector = np.array(query_vector).astype("float32").reshape(1, -1)

        distances, indices = self.index.search(query_vector, k)
        #distance between the query and each chunk, the chunks that are more relevant 
        # k == top-k retreval check in stanford course, top 3 highest chunks
        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results
    
# Storage
# embeddings saved in FAISS index
# 🔵 Search
# input question → vector
# FAISS finds closest chunks 

# FAISS does:
# “Find the closest vectors to a query vector”, whole chunks are returned if close enough
# “Which chunks are closest in meaning to this question?”