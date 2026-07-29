"""Retrieval-augmented answer generation over a video's vectorstore.

Retrieval is hybrid: dense vector similarity (Chroma/embeddings) catches
semantic matches ("explanation of overfitting" matching a chunk that never
says the word "overfitting"), while BM25 keyword search catches exact-term
matches that embeddings sometimes blur past (names, acronyms, specific
numbers). Combining both with an EnsembleRetriever generally beats either
alone.
"""
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

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

# BM25 retrievers are cheap to rebuild (a few hundred chunks, no embeddings
# involved) but there's no reason to redo it every question — cache per video.
_bm25_cache: dict[str, BM25Retriever] = {}


def get_hybrid_retriever(db: Chroma, video_id: str, k: int = None):
    k = k or config.RETRIEVAL_K
    vector_retriever = db.as_retriever(search_kwargs={"k": k})

    bm25 = _bm25_cache.get(video_id)
    if bm25 is None:
        all_docs = db.get()["documents"]
        bm25 = BM25Retriever.from_texts(all_docs)
        _bm25_cache[video_id] = bm25
    bm25.k = k

    # weights favor semantic match slightly, since transcript language is
    # conversational and exact keyword overlap with a question is less common
    return EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.6, 0.4])


def answer_question(db: Chroma, video_id: str, query: str, k: int = None) -> dict:
    retriever = get_hybrid_retriever(db, video_id, k=k)
    docs = retriever.invoke(query)
    context = " ".join(d.page_content for d in docs)

    prompt = PROMPT_TEMPLATE.format(context=context[:3000], question=query)
    response = _model.generate_content(prompt)

    return {
        "answer": response.text,
        "context_chunks": [d.page_content for d in docs],
    }
