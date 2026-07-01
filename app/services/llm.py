from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(question: str, chunks: list[str]):
    context = "\n\n".join(chunks)# prepares the chunks for the model

    prompt = f"""
You are a helpful assistant. Use ONLY the context below.

Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""
#prompt given to the model, we are telling it “don’t guess” “use only this info”

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )#the models answer

    return response.choices[0].message.content
#return the answer of the model

# joins FAISS chunks
# builds prompt
# sends to LLM
# returns final answer