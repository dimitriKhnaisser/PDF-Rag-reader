from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]):
    return model.encode(texts)

# loads a small AI model locally
# converts text → vectors (numbers)
# prepares data for FAISS search