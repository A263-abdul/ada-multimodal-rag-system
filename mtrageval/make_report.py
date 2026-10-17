"""Write the Task A submission, validate all submissions with the official format checker,
and fill the results tables into mtrageval/README.md.

python -m mtrageval.make_report
"""
import json
import subprocess
import sys

from mtrageval.common import DATA, OFFICIAL_COLLECTION, ROOT, RUNS_DIR, load_tasks, write_jsonl

SUBMIT_DIR = DATA / "submissions"
README = ROOT / "mtrageval" / "README.md"


def write_inputs_and_taskA():
    test = load_tasks("test")
    top10 = json.load(open(RUNS_DIR / "test" / "final_top10.json"))
    inputs, taskA = [], []
    for t in test:
        base = {"conversation_id": t["conversation_id"], "task_id": t["task_id"],
                "Collection": OFFICIAL_COLLECTION[t["domain"]],
                "input": [{"speaker": m["speaker"], "text": m["text"]} for m in t["input"]]}
        inputs.append(base)
        taskA.append({**base, "contexts": top10.get(t["task_id"], [])})
    write_jsonl(SUBMIT_DIR / "test_input.jsonl", inputs)
    write_jsonl(SUBMIT_DIR / "taskA_test.jsonl", taskA)


def check_formats() -> list[str]:
    out = []
    for task, mode in (("A", "retrieval_taska"), ("B", "generation_taskb"), ("C", "rag_taskc")):
        res = subprocess.run([sys.executable, str(ROOT / "scripts/evaluation/format_checker.py"),
                              "--input_file", str(SUBMIT_DIR / "test_input.jsonl"),
                              "--prediction_file", str(SUBMIT_DIR / f"task{task}_test.jsonl"), "--mode", mode],
                             capture_output=True, text=True)
        verdict = [l for l in res.stdout.splitlines() if "Format is valid" in l or "issues" in l or "Error" in l]
        out.append(f"Task {task} ({mode}): {verdict[-1].strip() if verdict else res.stdout.strip()[-200:]}")
    return out


def tables() -> str:
    a = json.load(open(DATA / "results_taskA.json"))
    bc = json.load(open(DATA / "results_taskBC.json"))
    lines = ["### Task A — Retrieval", "",
             f"Chosen settings (tuned on dev): `{json.dumps(a['config'])}`", "",
             "| Split | System | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | R@5 | R@10 |",
             "|---|---|---|---|---|---|---|---|"]
    names = {"bm25_last": "BM25, last turn", "bm25_tuned": "BM25, tuned multi-turn query",
             "final": "**+ cross-encoder rerank (final)**"}
    for split in ("dev", "test"):
        for key, name in names.items():
            m = a["results"][split][key]
            lines.append(f"| {split} | {name} | " + " | ".join(
                f"{m[k]:.3f}" for k in ("nDCG@1", "nDCG@3", "nDCG@5", "nDCG@10", "R@5", "R@10")) + " |")
    lines += ["", "Paper baselines on MTRAG (nDCG@5): BM25 last turn 0.18, BM25 rewrite 0.22, "
              "BGE-base rewrite 0.34, Elser rewrite 0.48.", ""]

    metrics = ("Recall", "RougeL", "K-Prec", "Answerability-Acc", "IDK-rate")
    for task, title in (("B", "Task B — Generation with reference passages"), ("C", "Task C — RAG (retrieved top-5 passages)")):
        r = bc[task]
        lines += [f"### {title}", "", f"Answerability threshold (CV on dev): {r['threshold']:.2f}", "",
                  "| Split | System | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
        for split in ("dev", "test"):
            for key, name in (("always_answer", "always answer"), ("with_classifier", "**+ answerability classifier**")):
                m = r[split][key]
                lines.append(f"| {split} | {name} | " + " | ".join(f"{m[k]:.3f}" for k in metrics) + " |")
        lines += ["", "Test, by answerability label (with classifier):", "",
                  "| Label | n | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
        for lab, m in sorted(r["test"]["by_label"].items()):
            lines.append(f"| {lab} | {m['n']} | " + " | ".join(f"{m[k]:.3f}" for k in metrics) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    write_inputs_and_taskA()
    checks = check_formats()
    body = tables() + "\n### Submission format check (official `format_checker.py`)\n\n" + \
        "\n".join(f"- {c}" for c in checks) + "\n"
    text = README.read_text()
    start = text.index("## Results") + len("## Results\n\n")
    end = text.index("## Evaluation notes")
    README.write_text(text[:start] + body + "\n" + text[end:])
    print(body)


if __name__ == "__main__":
    main()
