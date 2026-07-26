"""Retrieval-augmented answer generation over a video's vectorstore."""
import google.generativeai as genai
from langchain_chroma import Chroma

from . import config

genai.configure(api_key=config.GOOGLE_API_KEY)
_model = genai.GenerativeModel(config.GEMINI_MODEL)

PROMPT_TEMPLATE = """You are an expert AI tutor explaining concepts from a lecture.

Use ONLY the context below to answer. If the context does not contain enough
information, say so honestly instead of guessing.

Context:
{context}

Question:
{question}

Answer clearly in 3-5 sentences:"""


def answer_question(db: Chroma, query: str, k: int = None) -> dict:
    k = k or config.RETRIEVAL_K
    docs = db.similarity_search(query, k=k)
    context = " ".join(d.page_content for d in docs)

    prompt = PROMPT_TEMPLATE.format(context=context[:3000], question=query)
    response = _model.generate_content(prompt)

    return {
        "answer": response.text,
        "context_chunks": [d.page_content for d in docs],
    }
