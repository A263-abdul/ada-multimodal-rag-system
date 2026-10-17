# MTRAGEval (SemEval-2026 Task 8): baseline system

A first, fully local system for all three subtasks of
[SemEval-2026 Task 8 — MTRAGEval](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/).
It runs on a laptop (Apple M2, 16 GB) with no external API.

| Split | Data | Use |
|---|---|---|
| dev | `mtrag-human/` (MTRAG, 842 tasks, 777 with qrels) | tuning / training |
| test | `mtragun-human/` (MTRAG-UN, 507 tasks, 332 with qrels) | final evaluation only |

## System

**Task A — Retrieval** ([retrieval.py](retrieval.py), [bm25.py](bm25.py))
1. BM25 (`bm25s`, English stemming and stopwords) over each domain's passage corpus.
   The query is the last user turn up-weighted (repeated) plus the previous user turns. The
   number of previous turns and the repeat count were tuned on dev.
2. The BM25 top 50 are reranked with the cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   Its scores are fused with BM25 scores; the fusion weight and reranker query were tuned on dev.
3. The top 10 passages are output.

**Task B — Generation with reference passages** ([generation.py](generation.py))
1. The passages are split into sentences, and each sentence is scored against the conversation
   (last two user turns) with the same cross-encoder.
2. The response is the top 3 sentences (≤ 90 words), so it is extractive and stays faithful to the passages.
3. An **answerability classifier** (logistic regression on sentence-score features, trained on the dev
   answerability labels; threshold chosen by 5-fold cross-validation on dev) decides whether to answer
   or reply *"I'm sorry, but I don't have the answer to your question."*

**Task C — Full RAG**: Task A's top 5 passages are passed to the Task B generator.

## How to run

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch sentence-transformers "bm25s[full]" pytrec-eval-terrier scikit-learn rouge-score
mkdir -p data/corpora && for f in corpora/passage_level/*.zip; do unzip -o -q $f -d data/corpora/; done

.venv/bin/python -m mtrageval.retrieval    # Task A: tune on dev, evaluate dev + test
.venv/bin/python -m mtrageval.generation   # Tasks B + C: train classifier on dev, evaluate
.venv/bin/python -m mtrageval.make_report  # print results tables, validate submission files
```

Outputs: `data/results_task*.json` (metrics), `data/submissions/task{A,B,C}_test.jsonl`
(official submission format, validated with `scripts/evaluation/format_checker.py`).

## Results

### Task A — Retrieval

Chosen settings (tuned on dev): `{"bm25_prev": 2, "bm25_repeat_last": 3, "ce_prev": 0, "w_bm25": 0.0, "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "first_stage_k": 50}`

| Split | System | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|
| dev | BM25, last turn | 0.179 | 0.174 | 0.195 | 0.224 | 0.218 | 0.286 |
| dev | BM25, tuned multi-turn query | 0.203 | 0.194 | 0.215 | 0.247 | 0.240 | 0.319 |
| dev | **+ cross-encoder rerank (final)** | 0.346 | 0.311 | 0.330 | 0.359 | 0.345 | 0.413 |
| test | BM25, last turn | 0.265 | 0.246 | 0.277 | 0.321 | 0.312 | 0.418 |
| test | BM25, tuned multi-turn query | 0.265 | 0.285 | 0.315 | 0.357 | 0.363 | 0.470 |
| test | **+ cross-encoder rerank (final)** | 0.395 | 0.357 | 0.393 | 0.433 | 0.431 | 0.523 |

Paper baselines on MTRAG (nDCG@5): BM25 last turn 0.18, BM25 rewrite 0.22, BGE-base rewrite 0.34, Elser rewrite 0.48.

### Task B — Generation with reference passages

Answerability threshold (CV on dev): 0.10

| Split | System | Recall | RougeL | K-Prec | Answerability-Acc | IDK-rate |
|---|---|---|---|---|---|---|
| dev | always answer | 0.356 | 0.291 | 0.884 | 0.999 | 0.078 |
| dev | **+ answerability classifier** | 0.356 | 0.291 | 0.884 | 0.999 | 0.078 |
| test | always answer | 0.330 | 0.267 | 0.913 | 0.911 | 0.256 |
| test | **+ answerability classifier** | 0.330 | 0.267 | 0.913 | 0.911 | 0.256 |

Test, by answerability label (with classifier):

| Label | n | Recall | RougeL | K-Prec | Answerability-Acc | IDK-rate |
|---|---|---|---|---|---|---|
| ANSWERABLE | 285 | 0.385 | 0.305 | 0.885 | 1.000 | 0.000 |
| PARTIAL | 47 | 0.381 | 0.282 | 0.929 | 1.000 | 0.000 |
| UNANSWERABLE | 97 | 0.271 | 0.269 | 1.000 | 1.000 | 1.000 |
| UNDERSPECIFIED | 78 | 0.174 | 0.116 | 0.901 | 0.423 | 0.423 |

### Task C — RAG (retrieved top-5 passages)

Answerability threshold (CV on dev): 0.45

| Split | System | Recall | RougeL | K-Prec | Answerability-Acc | IDK-rate |
|---|---|---|---|---|---|---|
| dev | always answer | 0.261 | 0.194 | 0.807 | 0.923 | 0.000 |
| dev | **+ answerability classifier** | 0.216 | 0.175 | 0.873 | 0.702 | 0.297 |
| test | always answer | 0.269 | 0.169 | 0.809 | 0.655 | 0.000 |
| test | **+ answerability classifier** | 0.229 | 0.167 | 0.880 | 0.647 | 0.300 |

Test, by answerability label (with classifier):

| Label | n | Recall | RougeL | K-Prec | Answerability-Acc | IDK-rate |
|---|---|---|---|---|---|---|
| ANSWERABLE | 285 | 0.246 | 0.197 | 0.869 | 0.779 | 0.221 |
| PARTIAL | 47 | 0.180 | 0.143 | 0.897 | 0.681 | 0.319 |
| UNANSWERABLE | 97 | 0.251 | 0.156 | 0.904 | 0.443 | 0.443 |
| UNDERSPECIFIED | 78 | 0.172 | 0.086 | 0.879 | 0.397 | 0.397 |

### Submission format check (official `format_checker.py`)

- Task A (retrieval_taska): Format is valid for the eval script.
- Task B (generation_taskb): Format is valid for the eval script.
- Task C (rag_taskc): Format is valid for the eval script.

## Evaluation notes and limitations

- Retrieval is scored with `pytrec_eval`, mirroring `scripts/evaluation/run_retrieval_eval.py`
  (mean over queries that have qrels). That script's qrels paths point to folders not in the repo,
  so qrels are read from `mtrag-human/retrieval_tasks/*/qrels/dev.tsv` and `mtragun-human/retrieval_tasks/qrels/`.
- The official generation score is the harmonic mean of RL_F, RB_llm and RB_alg. RB_llm and RL_F need
  an LLM judge, and RB_alg uses BERTScore with DeBERTa-xlarge. Neither was run in this first version.
  We report the official **lexical** metrics instead (Recall and ROUGE-L against the reference answer,
  K-Prec against the passages, same text normalization as `run_algorithmic.py`) plus answerability accuracy.
- The generator is extractive. An LLM generator would be the natural next step for Tasks B/C.
- The dev set has no UNDERSPECIFIED labels, so the classifier treats those like unanswerable questions.
