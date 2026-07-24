# NTRS Keyword Normalization — Benchmark Methodology

A short note on how NTRS queries are formed during benchmark generation, and why.

## Shared-query fairness principle

For every atom, a single query is produced by the shared `QueryBuilder` and used as the common starting point for **both** retrieval backends. Google receives that query as-is; NTRS receives a deterministically normalized form of the **same** query. Because both backends originate from one generated query, the benchmark compares the *retrieval systems* against each other, not two independently generated queries.

## Why NTRS needs deterministic keyword normalization

Google natively handles full natural-language queries. NTRS is fundamentally a keyword-based search engine that **AND-matches** every term, so a long natural-language query matches almost nothing and would make NTRS look empty for the wrong reason. To keep the comparison about evidence quality rather than query formatting, we deterministically normalize the shared query into a short keyword query before NTRS retrieval.

## Normalization philosophy

The normalization preserves the most informative scientific retrieval anchors — mission names, instruments, experiments, celestial bodies, named people and places, and scientific concepts — while demoting generic contextual language (reporting verbs, vague qualifiers, and corpus-implicit terms). It is fully deterministic and adds **no additional LLM calls**. The objective is *faithful representation of the original scientific query*, not maximizing NTRS retrieval.

## Methodology diagram

```
Atom
   │
   ▼
QueryBuilder (shared)
   │
   ├──────────────► Google Retriever
   │
   ▼
Deterministic Retrieval-Anchor Normalization
   │
   ▼
NTRS Retriever
```

## A note on honest corpus limitations

Some topics are simply not represented in NTRS (for example, historical or outreach-oriented facts). Such gaps are expected and left intact: an absent topic should remain absent rather than being optimized away. When NTRS returns nothing, we want confidence that it failed because the corpus lacks appropriate documents — not because normalization discarded the most important scientific concepts.
