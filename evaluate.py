"""
Optional: run the same evaluation suite from the original notebook
(ROUGE, BLEU, BERTScore, context precision/recall, hallucination rate)
against a video you've already processed.

This is intentionally kept OUT of the main app / requirements.txt because
rouge-score / bert-score / nltk / matplotlib are heavy and only needed for
offline evaluation, not for serving the RAG app itself.

Usage:
    pip install rouge-score bert-score nltk scikit-learn matplotlib pandas
    python evaluate.py --video-id <id> --questions questions.txt
"""
import argparse
import re
import numpy as np
import pandas as pd

from rag import ingest, qa


def detect_lang(text: str) -> str:
    if any(0x0B80 <= ord(c) <= 0x0BFF for c in text):
        return "ta"
    if any(0x0900 <= ord(c) <= 0x097F for c in text):
        return "hi"
    return "en"


def compute_rouge(answer: str, reference: str) -> dict:
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, answer)
    return {
        "ROUGE-1": round(s["rouge1"].fmeasure, 4),
        "ROUGE-2": round(s["rouge2"].fmeasure, 4),
        "ROUGE-L": round(s["rougeL"].fmeasure, 4),
    }


def compute_bleu(answer: str, reference: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method4
    return round(float(sentence_bleu(
        [reference.lower().split()], answer.lower().split(),
        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth,
    )), 4)


def compute_bertscore(answer: str, reference: str, lang: str = "en") -> dict:
    from bert_score import score as bert_score_fn
    model_type = "bert-base-multilingual-cased" if lang != "en" else "bert-base-uncased"
    P, R, F1 = bert_score_fn([answer], [reference], model_type=model_type, lang=lang, verbose=False)
    return {
        "BERTScore-P": round(P.mean().item(), 4),
        "BERTScore-R": round(R.mean().item(), 4),
        "BERTScore-F1": round(F1.mean().item(), 4),
    }


def compute_context_pr(answer: str, chunks: list) -> dict:
    a_tok = set(answer.lower().split())
    relevant = sum(
        1 for c in chunks
        if len(a_tok & set(c.lower().split())) / max(len(set(c.lower().split())), 1) > 0.05
    )
    precision = relevant / len(chunks) if chunks else 0
    all_ctx = set(" ".join(chunks).lower().split())
    recall = len(a_tok & all_ctx) / max(len(a_tok), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"Context Precision": round(precision, 4), "Context Recall": round(recall, 4), "Context F1": round(f1, 4)}


def compute_hallucination(answer: str, context: str, embed_fn) -> dict:
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    THRESH = 0.55
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.split()) > 4]
    if not sents:
        return {"Hallucination Rate": 0.0, "Grounded Rate": 1.0}
    ctx_emb = np.array(embed_fn(context)).reshape(1, -1)
    flagged = [s for s in sents if cos_sim(np.array(embed_fn(s)).reshape(1, -1), ctx_emb)[0][0] < THRESH]
    h = len(flagged) / len(sents)
    return {"Hallucination Rate": round(h, 4), "Grounded Rate": round(1 - h, 4)}


def evaluate_one(db, video_id, embedding, query: str) -> dict:
    result = qa.answer_question(db, video_id, query)
    answer = result["answer"]
    chunks = result["context_chunks"]
    context = " ".join(chunks)
    ref = context  # no gold reference available; falls back to retrieved context
    lang = detect_lang(query)
    embed_fn = embedding.embed_query

    return {
        "Question": query,
        "Language": lang,
        "Answer": answer,
        **compute_rouge(answer, ref),
        "BLEU": compute_bleu(answer, ref),
        **compute_bertscore(answer, ref, lang),
        **compute_context_pr(answer, chunks),
        **compute_hallucination(answer, context, embed_fn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--questions", required=True, help="Text file, one question per line")
    args = parser.parse_args()

    db = ingest.load_vectorstore(args.video_id)
    embedding = ingest.get_embedding_model()

    with open(args.questions, encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    rows = [evaluate_one(db, args.video_id, embedding, q) for q in questions]
    df = pd.DataFrame(rows)

    metric_cols = [c for c in df.columns if c not in ("Question", "Language", "Answer")]
    print("\nPER-QUESTION RESULTS")
    print(df[["Question"] + metric_cols].to_string(index=False))
    print("\nAGGREGATE (mean)")
    print(df[metric_cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
