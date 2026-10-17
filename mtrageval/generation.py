"""Tasks B and C: extractive response generation + an answerability classifier trained on dev.

For each task the passages are split into sentences, each sentence is scored against the conversation
with a cross-encoder, and the top sentences form the response. A logistic-regression classifier
(trained on the MTRAG dev labels) decides whether to answer or to reply "I don't know".

python -m mtrageval.generation      (run after mtrageval.retrieval)
"""
import json
import re
from collections import Counter

import joblib
import numpy as np
import torch
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from mtrageval.common import DATA, OFFICIAL_COLLECTION, RUNS_DIR, load_tasks, user_turns, write_jsonl
from mtrageval.retrieval import RERANKER

IDK = "I'm sorry, but I don't have the answer to your question."
MAX_SENTENCES = 3
MAX_WORDS = 90
SUBMIT_DIR = DATA / "submissions"


def answerable_label(task) -> int:
    labels = task.get("Answerability") or task.get("answerability") or []
    return int(any(l in ("ANSWERABLE", "PARTIAL") for l in labels))


def split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sents if 5 <= len(s.split()) <= 80]


def query_of(task) -> str:
    turns = user_turns(task)
    return " ".join(turns[-2:][::-1])  # last user turn first, then the previous one


def score_task(model, task, contexts):
    """Scores every candidate sentence. Returns (sentences sorted by score, features)."""
    q = query_of(task)
    sents = [s for c in contexts for s in split_sentences(c["text"])]
    if not sents:
        return [], np.array([-12.0, -12.0, -12.0, 0.0, len(q.split()), len(user_turns(task))])
    scores = model.predict([(q, s) for s in sents], batch_size=64, show_progress_bar=False)
    order = np.argsort(-scores)
    ranked = [(sents[i], float(scores[i])) for i in order]
    top = scores[order]
    feats = np.array([top[0], top[:3].mean(), top[:10].mean(), np.log1p(len(contexts)),
                      len(q.split()), len(user_turns(task))])
    return ranked, feats


def compose(ranked) -> str:
    out, n_words, seen = [], 0, set()
    for s, _ in ranked:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        if out and n_words + len(s.split()) > MAX_WORDS:
            break
        out.append(s)
        n_words += len(s.split())
        if len(out) == MAX_SENTENCES:
            break
    return " ".join(out)


# ---------------------------------------------------------------- lexical metrics (as in run_algorithmic.py)

def _norm(t):
    t = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", "", t.lower()))
    return t.split()


def recall(pred, target):
    p, t = _norm(pred), _norm(target)
    common = sum((Counter(p) & Counter(t)).values())
    return common / len(t) if common else 0.0


def k_prec(pred, passage):
    p, t = _norm(pred), _norm(passage)
    common = sum((Counter(p) & Counter(t)).values())
    return common / len(p) if common else 0.0


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def evaluate_generation(tasks, preds, contexts_by_task) -> dict:
    rows = []
    for t in tasks:
        pred, target = preds[t["task_id"]], t["targets"][0]["text"]
        ctx = contexts_by_task.get(t["task_id"], [])
        gold_ans, pred_ans = answerable_label(t), int(pred != IDK)
        rows.append({
            "Recall": recall(pred, target),
            "RougeL": _rouge.score(target, pred)["rougeL"].fmeasure,
            "K-Prec": max((k_prec(pred, c["text"]) for c in ctx), default=0.0) if pred_ans else 1.0,
            "Answerability-Acc": float(gold_ans == pred_ans),
            "IDK-rate": float(not pred_ans),
        })
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["n"] = len(rows)
    return agg


# ---------------------------------------------------------------- main

def featurize(model, tasks, contexts_by_task):
    ranked, feats = {}, []
    for t in tasks:
        r, f = score_task(model, t, contexts_by_task.get(t["task_id"], []))
        ranked[t["task_id"]] = r
        feats.append(f)
    return ranked, np.vstack(feats)


def predict(tasks, ranked, probs, threshold):
    return {t["task_id"]: (compose(ranked[t["task_id"]]) if p >= threshold and ranked[t["task_id"]] else IDK)
            for t, p in zip(tasks, probs)}


def main():
    dev, test = load_tasks("dev"), load_tasks("test")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CrossEncoder(RERANKER, device=device, max_length=256)
    top10 = {s: json.load(open(RUNS_DIR / s / "final_top10.json")) for s in ("dev", "test")}

    settings = {
        "B": {s: {t["task_id"]: t["contexts"] for t in ts} for s, ts in (("dev", dev), ("test", test))},
        "C": {s: {tid: ctx[:5] for tid, ctx in top10[s].items()} for s in ("dev", "test")},
    }
    results = {}
    for task_name, ctxs in settings.items():
        print(f"== Task {task_name}: scoring sentences ==", flush=True)
        dev_ranked, X_dev = featurize(model, dev, ctxs["dev"])
        test_ranked, X_test = featurize(model, test, ctxs["test"])
        y_dev = np.array([answerable_label(t) for t in dev])

        # Train the answerability classifier on dev; pick the threshold by cross-validated balanced accuracy
        # (unanswerable questions are 7% of dev but 35% of test, so a prior-independent criterion is used).
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        cv_probs = cross_val_predict(clf, X_dev, y_dev, cv=5, method="predict_proba")[:, 1]
        best_thr, best = 0.5, -1
        for thr in np.linspace(0.1, 0.9, 17):
            pred = predict(dev, dev_ranked, cv_probs, thr)
            pred_ans = np.array([int(pred[t["task_id"]] != IDK) for t in dev])
            bal = np.mean([np.mean(pred_ans[y_dev == c] == c) for c in (0, 1)])
            if bal > best:
                best, best_thr = bal, thr
        clf.fit(X_dev, y_dev)
        (DATA / "models").mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": clf, "threshold": float(best_thr)}, DATA / "models" / f"answerability_{task_name}.joblib")
        print(f"   threshold={best_thr:.2f}", flush=True)

        res = {"threshold": float(best_thr)}
        for split, tasks, ranked, probs in (("dev", dev, dev_ranked, cv_probs),
                                            ("test", test, test_ranked, clf.predict_proba(X_test)[:, 1])):
            preds = predict(tasks, ranked, probs, best_thr)
            always = {t["task_id"]: compose(ranked[t["task_id"]]) or IDK for t in tasks}
            res[split] = {"always_answer": evaluate_generation(tasks, always, ctxs[split]),
                          "with_classifier": evaluate_generation(tasks, preds, ctxs[split])}
            # Per answerability label on test.
            if split == "test":
                by = {}
                for t in tasks:
                    lab = (t.get("answerability") or ["?"])[0]
                    by.setdefault(lab, []).append(t)
                res[split]["by_label"] = {lab: evaluate_generation(ts, preds, ctxs[split]) for lab, ts in by.items()}
                write_submission(task_name, tasks, preds, ctxs[split])
            print(f"   [{split}] always-answer  ", fmt(res[split]["always_answer"]))
            print(f"   [{split}] + classifier   ", fmt(res[split]["with_classifier"]), flush=True)
        results[task_name] = res
    with open(DATA / "results_taskBC.json", "w") as f:
        json.dump(results, f, indent=2)


def fmt(m):
    return " ".join(f"{k}={v:.4f}" for k, v in m.items() if k != "n") + f" n={m['n']}"


def write_submission(task_name, tasks, preds, ctxs):
    rows = []
    for t in tasks:
        row = {"conversation_id": t["conversation_id"], "task_id": t["task_id"],
               "Collection": OFFICIAL_COLLECTION[t["domain"]],
               "input": [{"speaker": m["speaker"], "text": m["text"]} for m in t["input"]],
               "contexts": [{"document_id": c["document_id"], "text": c["text"], "score": c.get("score", 1.0)}
                            for c in ctxs.get(t["task_id"], [])[:10]],  # format checker allows at most 10
               "predictions": [{"text": preds[t["task_id"]]}]}
        rows.append(row)
    write_jsonl(SUBMIT_DIR / f"task{task_name}_test.jsonl", rows)


if __name__ == "__main__":
    main()
