from app.services.embeddings import embed_texts
import numpy as np


question = "Give some projects that Dimitri has worked on"

texts = [
    "PDF RAG system project",
    "Image captioning project",
    "Machine learning project",
    "Robotics project",
    "Job Board API project",
    "Education and academic background",
    "Work experience",
]


vectors = embed_texts([question] + texts)


question_vector = np.array(vectors[0])


for i, text in enumerate(texts):

    vector = np.array(vectors[i + 1])

    similarity = np.dot(question_vector, vector) / (
        np.linalg.norm(question_vector) *
        np.linalg.norm(vector)
    )

    print(f"{text}: {similarity:.4f}")