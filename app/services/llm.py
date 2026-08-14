import os
from groq import Groq


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(question: str, retrieved_chunks: list):

    # Build context from retrieved chunks
    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"Source: {chunk['source']}\n"
            f"Content:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)


    prompt = f"""
You are a helpful assistant answering questions about the provided documents.

Use ONLY the information contained in the context below.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents."

For each important piece of information, mention the source PDF.

Context:
{context}

Question:
{question}

Answer:
"""


    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content
#return the answer of the model






# joins FAISS chunks
# builds prompt
# sends to LLM
# returns final answer