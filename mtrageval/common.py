"""Shared data loading, query construction and retrieval scoring for MTRAGEval."""
import csv
import json
from pathlib import Path

import pytrec_eval

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPORA_DIR = DATA / "corpora"
RUNS_DIR = DATA / "runs"

DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]

# Official collection names used by the evaluation scripts / submission format.
OFFICIAL_COLLECTION = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "govt": "mt-rag-govt-elser-512-100-20240611",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "cloud": "mt-rag-ibmcloud-elser-512-100-20240502",
}

SPLITS = {
    # MTRAG human data: training/dev for the shared task.
    "dev": {
        "tasks": ROOT / "mtrag-human/generation_tasks/reference.jsonl",
        "qrels": {d: ROOT / f"mtrag-human/retrieval_tasks/{d}/qrels/dev.tsv" for d in DOMAINS},
    },
    # MTRAG-UN: the SemEval evaluation benchmark.
    "test": {
        "tasks": ROOT / "mtragun-human/generation_tasks/reference.jsonl",
        "qrels": {d: ROOT / f"mtragun-human/retrieval_tasks/qrels/{d}.tsv" for d in DOMAINS},
    },
}


def domain_of(collection: str) -> str:
    c = collection.lower()
    if "ibmcloud" in c or "cloud" in c:
        return "cloud"
    for d in ("clapnq", "fiqa", "govt"):
        if d in c:
            return d
    raise ValueError(f"unknown collection {collection!r}")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_tasks(split: str):
    """Tasks with a `domain` field added. Each task's `input` is the conversation up to the current user turn."""
    tasks = read_jsonl(SPLITS[split]["tasks"])
    for t in tasks:
        t["domain"] = domain_of(t["Collection"])
    return tasks


def load_qrels(split: str):
    qrels = {}
    for d, path in SPLITS[split]["qrels"].items():
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for qid, docid, score in reader:
                qrels.setdefault(qid, {})[docid] = int(score)
    return qrels


def iter_corpus(domain: str):
    with open(CORPORA_DIR / f"{domain}.jsonl", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def passage_text(doc) -> str:
    title, text = (doc.get("title") or "").strip(), doc["text"].strip()
    if title and not text.startswith(title):
        return f"{title}\n{text}"
    return text


# ---------------------------------------------------------------- queries

def user_turns(task):
    return [m["text"].strip() for m in task["input"] if m["speaker"] == "user"]


def last_turn(task) -> str:
    return user_turns(task)[-1]


def build_query(task, mode: str = "last", rewrites: dict | None = None) -> str:
    """Query strings for retrieval.

    last      - final user utterance only
    all_user  - all user utterances, most recent first
    rewrite   - an externally produced standalone rewrite (falls back to last)
    """
    if mode == "last":
        return last_turn(task)
    if mode == "all_user":
        return "\n".join(reversed(user_turns(task)))
    if mode == "rewrite":
        return (rewrites or {}).get(task["task_id"]) or last_turn(task)
    raise ValueError(mode)


# ---------------------------------------------------------------- scoring

def evaluate_run(run: dict, qrels: dict, ks=(1, 3, 5, 10)) -> tuple[dict, dict]:
    """Mirror of scripts/evaluation/run_retrieval_eval.py: mean over queries that have qrels.

    run: {qid: {docid: score}}. Returns (aggregate, per_query).
    """
    run = {q: d for q, d in run.items() if q in qrels}
    measures = {"ndcg_cut." + ",".join(map(str, ks)), "recall." + ",".join(map(str, ks))}
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in run}, measures)
    per_q = evaluator.evaluate(run)
    agg = {}
    for k in ks:
        agg[f"nDCG@{k}"] = sum(s[f"ndcg_cut_{k}"] for s in per_q.values()) / max(len(per_q), 1)
        agg[f"R@{k}"] = sum(s[f"recall_{k}"] for s in per_q.values()) / max(len(per_q), 1)
    agg["n"] = len(per_q)
    return agg, per_q


def report(name: str, run: dict, qrels: dict, tasks=None) -> dict:
    agg, per_q = evaluate_run(run, qrels)
    line = f"{name:<40} n={agg['n']:<4} " + " ".join(
        f"{m}={agg[m]:.4f}" for m in ("nDCG@1", "nDCG@3", "nDCG@5", "nDCG@10", "R@5", "R@10"))
    if tasks is not None:
        by_dom = {}
        for t in tasks:
            if t["task_id"] in per_q:
                by_dom.setdefault(t["domain"], []).append(per_q[t["task_id"]]["ndcg_cut_5"])
        line += "  | nDCG@5 " + " ".join(f"{d}={sum(v)/len(v):.3f}" for d, v in sorted(by_dom.items()))
    print(line, flush=True)
    return agg


def save_run(path, run: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(run, f)


def load_run(path) -> dict:
    with open(path) as f:
        return json.load(f)
