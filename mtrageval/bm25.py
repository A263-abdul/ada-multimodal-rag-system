"""BM25 (bm25s) indexing and retrieval per domain.

python -m mtrageval.bm25 --split dev --query last
"""
import argparse

import bm25s
import Stemmer

from mtrageval.common import (DATA, DOMAINS, RUNS_DIR, build_query, iter_corpus, load_qrels,
                              load_tasks, passage_text, read_jsonl, report, save_run)

INDEX_DIR = DATA / "index" / "bm25"
stemmer = Stemmer.Stemmer("english")


def build_index(domain: str):
    docs = list(iter_corpus(domain))
    ids = [d["_id"] for d in docs]
    tokens = bm25s.tokenize([passage_text(d) for d in docs], stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(tokens, show_progress=False)
    retriever.save(INDEX_DIR / domain, corpus=[{"id": i} for i in ids])
    return retriever


def load_index(domain: str):
    if not (INDEX_DIR / domain).exists():
        print(f"building BM25 index for {domain}", flush=True)
        build_index(domain)
    return bm25s.BM25.load(INDEX_DIR / domain, load_corpus=True)


def retrieve(tasks, queries: dict, k: int = 100) -> dict:
    """queries: {task_id: query string}. Returns run {task_id: {docid: score}}."""
    run = {}
    for domain in DOMAINS:
        dom_tasks = [t for t in tasks if t["domain"] == domain]
        if not dom_tasks:
            continue
        retriever = load_index(domain)
        q_tokens = bm25s.tokenize([queries[t["task_id"]] for t in dom_tasks], stopwords="en",
                                  stemmer=stemmer, show_progress=False)
        docs, scores = retriever.retrieve(q_tokens, k=k, show_progress=False)
        for t, drow, srow in zip(dom_tasks, docs, scores):
            run[t["task_id"]] = {d["id"]: float(s) for d, s in zip(drow, srow) if s > 0}
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--query", default="last", help="last | all_user | rewrite:<path to rewrites jsonl>")
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    tasks = load_tasks(args.split)
    mode, _, path = args.query.partition(":")
    rewrites = {r["task_id"]: r["rewrite"] for r in read_jsonl(path)} if path else None
    queries = {t["task_id"]: build_query(t, mode, rewrites) for t in tasks}
    run = retrieve(tasks, queries, args.k)
    tag = mode if not path else f"rewrite-{path.split('/')[-1].removesuffix('.jsonl')}"
    save_run(RUNS_DIR / args.split / f"bm25_{tag}.json", run)
    report(f"bm25 [{tag}] {args.split}", run, load_qrels(args.split), tasks)


if __name__ == "__main__":
    main()
