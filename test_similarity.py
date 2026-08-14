import pickle
import numpy as np

from app.services.embeddings import embed_texts


with open("data/vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


question = "Give some projects that Dimitri has worked on"


# Get question embedding
question_vector = embed_texts([question])[0]


# Get embeddings for the project chunks
project_chunks = [chunks[8], chunks[9], chunks[10], chunks[11]]

project_texts = [chunk["text"] for chunk in project_chunks]

project_vectors = embed_texts(project_texts)


# Calculate cosine similarity
question_vector = np.array(question_vector)

for i, vector in enumerate(project_vectors):

    vector = np.array(vector)

    similarity = np.dot(question_vector, vector) / (
        np.linalg.norm(question_vector) *
        np.linalg.norm(vector)
    )

    print(
        f"Chunk {i + 9} similarity: {similarity:.4f}"
    )