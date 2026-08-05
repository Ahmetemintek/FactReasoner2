# NASA Retrieval Benchmark

How to reproduce the Google-vs-NTRS retrieval benchmark added alongside the NTRS backend.

## 1. Motivation

FactReasoner2 was extended with an NTRS retrieval backend so that claim verification can draw on the NASA Technical Reports Server — an authoritative scientific and engineering corpus — as an alternative to generic web search. Adding a new retrieval backend only has value if it can be shown to improve evidence retrieval for claim verification, so this benchmark compares Google and NTRS retrieval under identical conditions on the same claims. Because retrieval quality is otherwise hard to isolate from prompt or query differences, the benchmark is built to be fully reproducible: frozen queries, frozen contexts, one evaluation pipeline.

## 2. Benchmark Overview

```text
Atomic claim
    │
    ▼
Shared QueryBuilder
    │
    ├──────────────┐
    ▼              ▼
 Google         NTRS
(raw query)   (deterministic keyword normalization)
    │              │
    ▼              ▼
Frozen benchmark datasets (*_google.jsonl / *_ntrs.jsonl)
    │
    ▼
FactVerify evaluation (eval_nasa_benchmark.py)
```

## 3. Benchmark Suites

Two complementary benchmark suites are provided:

**Mission Knowledge** (`data/nasa_science-labeled*.jsonl`) — Apollo 11, Voyager 1, Perseverance. Mission narratives mixing historical facts with a few instrument/measurement claims.

**Technical Knowledge** (`data/nasa_technical-labeled*.jsonl`) — SHERLOC, PIXL, SuperCam, MOXIE, Mastcam-Z. Entirely instrumentation and measurement-technique claims on the Mars 2020 Perseverance rover.

Two suites rather than one because they stress different aspects of retrieval: Mission Knowledge primarily evaluates mission history and narrative facts, while Technical Knowledge focuses on instrument specifications, engineering details, and measurement methods within a single, technically dense domain.

## 4. Fair Comparison Principles

- **One shared, LLM-generated query per claim.** Both backends are evaluated against the same query — neither is tested with a query hand-tuned in its favor.
- **Deterministic NTRS keyword normalization.** NTRS receives a rule-based keyword form of the shared query (no additional LLM call); Google receives the query unchanged. See §6.
- **Frozen retrieval datasets.** Contexts are retrieved once and stored; evaluation performs no retrieval, so results are reproducible and Google/NTRS scores are always computed on the same evidence.
- **Identical verification pipeline.** Both backends are scored with the same verifier model, prompt, and pipeline configuration.
- **Retrieval corpus is the only variable.** Everything else — claims, gold labels, queries, verifier — is held constant, so any score difference is attributable to the evidence each corpus supplied. This isolates retrieval quality from differences in query formulation and verifier behavior.

## 5. Reproducing the Benchmark

Generate the frozen per-backend datasets from a base benchmark file:

```bash
export OPENAI_API_KEY=...   # or set in FactReasoner2/.env
export SERPER_API_KEY=...   # required for the google backend

python -m fact_reasoner.eval.build_ntrs_benchmark \
    --input_file data/nasa_technical-labeled.jsonl \
    --dataset_name nasa_technical-labeled \
    --output_dir data \
    --backend openai --model_id <supported-openai-model> \
    --services google,ntrs \
    --top_k 5 --ntrs_max_terms 3
```

This writes `data/nasa_technical-labeled_google.jsonl` and `data/nasa_technical-labeled_ntrs.jsonl`. Substitute `nasa_science-labeled` to regenerate the Mission Knowledge suite (uses `--ntrs_max_terms 4`, the default).

Evaluation never performs live retrieval — it operates exclusively on the frozen benchmark datasets generated in the previous step, so results are reproducible without repeating any Google or NTRS calls.

Evaluate a frozen dataset:

```bash
python -m fact_reasoner.eval.eval_nasa_benchmark \
    --input_file data/nasa_technical-labeled_google.jsonl \
    --output_dir results --dataset_name nasa_technical \
    --service_type google --backend openai --model_id <supported-openai-model> \
    --pipeline factverify
```

Run the same command against the `_ntrs.jsonl` file (with `--service_type ntrs`) to produce the comparison.

## 6. Related Documentation

See [`docs/examples/eval/ntrs_keyword_normalization.md`](ntrs_keyword_normalization.md) for the implementation details of the deterministic keyword-normalization strategy used for NTRS retrieval.

## Notes

This benchmark is designed to compare retrieval backends under identical verification conditions. It is not intended to measure the absolute capability of any particular language model.
