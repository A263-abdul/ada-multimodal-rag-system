"""Task A: BM25 first stage + cross-encoder reranking, with query settings tuned on dev.

python -m mtrageval.retrieval
"""
import json

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from mtrageval import bm25
from mtrageval.common import (DATA, DOMAINS, RUNS_DIR, iter_corpus, load_qrels, load_tasks,
                              passage_text, report, save_run, user_turns)

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FIRST_STAGE_K = 50


def make_query(task, n_prev: int, repeat_last: int) -> str:
    """Last user turn (repeated to up-weight it in BM25) plus the n_prev preceding user turns."""
    turns = user_turns(task)
    prev = turns[-1 - n_prev:-1] if n_prev else []
    return " ".join([turns[-1]] * repeat_last + prev)


def load_passages(ids: set) -> dict:
    out = {}
    for d in DOMAINS:
        for doc in iter_corpus(d):
            if doc["_id"] in ids:
                out[doc["_id"]] = {"text": doc["text"], "title": doc.get("title", ""), "full": passage_text(doc)}
    return out


def rerank(model, tasks, run, passages, n_prev: int) -> dict:
    """Returns {task_id: {docid: ce_score}} over the first-stage candidates."""
    out = {}
    for t in tasks:
        cands = list(run.get(t["task_id"], {}))[:FIRST_STAGE_K]
        if not cands:
            out[t["task_id"]] = {}
            continue
        q = make_query(t, n_prev, 1)
        scores = model.predict([(q, passages[c]["full"]) for c in cands], batch_size=64, show_progress_bar=False)
        out[t["task_id"]] = dict(zip(cands, map(float, scores)))
    return out


def fuse(bm25_run, ce_run, w: float) -> dict:
    """Combine min-max normalised BM25 and cross-encoder scores: w * bm25 + (1 - w) * ce."""
    fused = {}
    for qid, ce in ce_run.items():
        if not ce:
            fused[qid] = {}
            continue
        b = {d: bm25_run[qid].get(d, 0.0) for d in ce}

        def norm(x):
            v = np.array(list(x.values()))
            lo, hi = v.min(), v.max()
            return {k: (s - lo) / (hi - lo + 1e-9) for k, s in x.items()}

        nb, nc = norm(b), norm(ce)
        fused[qid] = dict(sorted(((d, w * nb[d] + (1 - w) * nc[d]) for d in ce), key=lambda x: -x[1]))
    return fused


def main():
    dev, test = load_tasks("dev"), load_tasks("test")
    dev_qrels, test_qrels = load_qrels("dev"), load_qrels("test")

    # 1) Tune the BM25 query on dev.
    print("== BM25 query tuning (dev) ==")
    best = None
    for n_prev in (0, 1, 2, 3):
        for rep in (1, 2, 3):
            if n_prev == 0 and rep > 1:
                continue
            run = bm25.retrieve(dev, {t["task_id"]: make_query(t, n_prev, rep) for t in dev}, k=FIRST_STAGE_K)
            agg = report(f"bm25 prev={n_prev} repeat_last={rep}", run, dev_qrels)
            if best is None or agg["R@10"] + agg["nDCG@5"] > best[0]:
                best = (agg["R@10"] + agg["nDCG@5"], n_prev, rep, run)
    _, bm_prev, bm_rep, dev_bm25 = best
    print(f"-> chose BM25 query: prev={bm_prev} repeat_last={bm_rep}")
    test_bm25 = bm25.retrieve(test, {t["task_id"]: make_query(t, bm_prev, bm_rep) for t in test}, k=FIRST_STAGE_K)

    # 2) Rerank with a cross-encoder; tune its query context and the fusion weight on dev.
    ids = {d for r in (dev_bm25, test_bm25) for v in r.values() for d in v}
    print(f"loading {len(ids)} candidate passages")
    passages = load_passages(ids)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CrossEncoder(RERANKER, device=device, max_length=384)

    print("== Reranker tuning (dev) ==")
    best = None
    for ce_prev in (0, 1):
        dev_ce = rerank(model, dev, dev_bm25, passages, ce_prev)
        for w in (0.0, 0.2, 0.4):
            agg = report(f"rerank ce_prev={ce_prev} w_bm25={w}", fuse(dev_bm25, dev_ce, w), dev_qrels)
            if best is None or agg["nDCG@5"] > best[0]:
                best = (agg["nDCG@5"], ce_prev, w)
    _, ce_prev, w = best
    config = {"bm25_prev": bm_prev, "bm25_repeat_last": bm_rep, "ce_prev": ce_prev, "w_bm25": w,
              "reranker": RERANKER, "first_stage_k": FIRST_STAGE_K}
    print("-> chose", config)

    # 3) Final runs with the chosen settings.
    results = {}
    for split, tasks, bm, qrels in (("dev", dev, dev_bm25, dev_qrels), ("test", test, test_bm25, test_qrels)):
        final = fuse(bm, rerank(model, tasks, bm, passages, ce_prev), w)
        save_run(RUNS_DIR / split / "bm25_tuned.json", bm)
        save_run(RUNS_DIR / split / "final.json", final)
        results[split] = {"bm25_last": report(f"[{split}] BM25 last turn",
                                              bm25.retrieve(tasks, {t["task_id"]: make_query(t, 0, 1) for t in tasks}),
                                              qrels, tasks),
                          "bm25_tuned": report(f"[{split}] BM25 tuned query", bm, qrels, tasks),
                          "final": report(f"[{split}] BM25 + rerank (final)", final, qrels, tasks)}
        # Passage texts for the top-10, used by Task A/C output files.
        top = {qid: [{"document_id": d, "score": s, "text": passages[d]["text"], "title": passages[d]["title"]}
                     for d, s in list(r.items())[:10]] for qid, r in final.items()}
        with open(RUNS_DIR / split / "final_top10.json", "w") as f:
            json.dump(top, f)
    with open(DATA / "results_taskA.json", "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
