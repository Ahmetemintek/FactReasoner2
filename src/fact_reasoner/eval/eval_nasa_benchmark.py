# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Evaluate the frozen NASA science benchmark datasets (one per retrieval
# backend) and report per-backend factuality metrics.
#
# This runner exists alongside eval_dataset.py because the latter requires an
# IBM-internal Mellea backend (mellea_ibm.rits) and, for the FactReasoner
# pipeline, the external `merlin` inference executable. This script uses a
# public Mellea backend (OpenAI by default) and the FactVerify / FactScore
# pipelines, which score atoms directly with the LLM and need no merlin.
#
# The benchmark datasets are frozen: contexts were retrieved once by
# build_ntrs_benchmark.py and stored in the .jsonl files. Evaluation therefore
# performs NO retrieval - `context_retriever` is deliberately None so that any
# accidental retrieval attempt fails loudly instead of silently querying a
# search backend and making the Google-vs-NTRS comparison irreproducible.
# `--service_type` is only a label used in the output filename.
#
# Example:
#   export OPENAI_API_KEY=...   # or set it in FactReasoner2/.env
#   python -m fact_reasoner.eval.eval_nasa_benchmark \
#       --input_file data/nasa_science-labeled_ntrs.jsonl \
#       --output_dir results --dataset_name nasa_science \
#       --service_type ntrs --model_id gpt-4o-mini --pipeline factverify

import os
import json
import argparse

from dotenv import load_dotenv

from mellea.backends import ModelOption

# Local imports
from fact_reasoner.baselines.factscore import FactScore
from fact_reasoner.baselines.factverify import FactVerify
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser

PIPELINES = {"factverify": FactVerify, "factscore": FactScore}


def make_backend(backend_name: str, model_id: str):
    """
    Create a public Mellea backend for the evaluation LLM.

    Args:
        backend_name: str
            The Mellea backend to use (currently "openai").
        model_id: str
            The model identifier passed to the backend.

    Returns:
        Backend: The constructed Mellea backend.
    """

    if backend_name == "openai":
        # OpenAIBackend reads OPENAI_API_KEY from the environment and also works
        # against any OpenAI-compatible endpoint via Mellea's configuration.
        from mellea.backends.openai import OpenAIBackend
        return OpenAIBackend(
            model_id=model_id,
            model_options={ModelOption.MAX_NEW_TOKENS: 4096},
        )

    raise ValueError(
        f"Unsupported evaluation backend: {backend_name!r}. Supported: 'openai'."
    )


def summarize(results: list) -> dict:
    """
    Aggregate per-response results into micro-averaged benchmark metrics.

    Precision/recall/F1 are computed over the atom-level confusion matrix
    against the gold labels, pooled across all responses.

    Args:
        results: list
            The per-response result dicts produced by the pipeline's score().

    Returns:
        dict: The aggregated metrics.
    """

    tp = sum(r.get("true_positive", 0) for r in results)
    tn = sum(r.get("true_negative", 0) for r in results)
    fp = sum(r.get("false_positive", 0) for r in results)
    fn = sum(r.get("false_negative", 0) for r in results)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    num_atoms = sum(r.get("num_atoms", 0) for r in results)
    accuracy = (tp + tn) / num_atoms if num_atoms else 0.0
    scores = [r["factuality_score"] for r in results if "factuality_score" in r]

    return {
        "num_responses": len(results),
        "num_atoms": num_atoms,
        "num_contexts": sum(r.get("num_contexts", 0) for r in results),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "avg_factuality_score": sum(scores) / len(scores) if scores else 0.0,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help="Path to a frozen benchmark dataset (jsonl with atoms and contexts)."
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default="results",
        help="Directory where the evaluation results are written."
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        default="nasa_science",
        help="Dataset name used in the output filename."
    )

    parser.add_argument(
        '--service_type',
        type=str,
        default="ntrs",
        help="Retrieval backend that produced the dataset (label only; no retrieval is performed)."
    )

    parser.add_argument(
        '--backend',
        type=str,
        default="openai",
        help="Public Mellea backend used by the evaluation pipeline (currently: openai)."
    )

    parser.add_argument(
        '--model_id',
        type=str,
        default="gpt-4o-mini",
        help="Model identifier passed to the evaluation backend."
    )

    parser.add_argument(
        '--pipeline',
        type=str,
        default="factverify",
        help="Factuality pipeline (factverify, factscore). Both avoid the merlin dependency."
    )

    args = parser.parse_args()

    if args.pipeline not in PIPELINES:
        raise ValueError(
            f"Unsupported pipeline: {args.pipeline!r}. Supported: {sorted(PIPELINES)}."
        )

    # Load OPENAI_API_KEY from FactReasoner2/.env if present.
    load_dotenv(override=True)

    backend = make_backend(args.backend, args.model_id)

    # The atom extractor and reviser are required to be non-None by build(),
    # but are never used here: the benchmark atoms are pre-defined and frozen
    # (has_atoms=True, revise_atoms=False).
    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)

    with open(args.input_file) as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(dataset)} records from {args.input_file}")

    output_filename = os.path.join(
        args.output_dir,
        "eval_{}_{}_{}_{}.jsonl".format(
            args.pipeline, args.service_type, args.dataset_name, args.model_id
        )
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support: skip responses that were already evaluated.
    evaluation_data = []
    if os.path.isfile(output_filename):
        with open(output_filename) as f:
            evaluation_data = [json.loads(line) for line in f if line.strip()]
    print(f"Found {len(evaluation_data)} existing evaluations in {output_filename}")

    processed_inputs = {res["query"] for res in evaluation_data if "query" in res}

    for record in dataset:
        if record["input"] in processed_inputs:
            print(f"Skipping already evaluated response: {record.get('topic')}")
            continue

        pipeline = PIPELINES[args.pipeline](
            backend=backend,
            atom_extractor=atom_extractor,
            atom_reviser=atom_reviser,
            context_retriever=None,  # frozen contexts: retrieval must not happen
        )

        pipeline.from_dict_with_contexts(record)
        pipeline.build(
            topic=record.get("topic"),
            has_atoms=True,
            has_contexts=True,
            revise_atoms=False,
        )

        results = pipeline.score()
        results["model_name"] = args.model_id
        results["service_type"] = args.service_type
        evaluation_data.append(results)

        with open(output_filename, "w") as f:
            for res in evaluation_data:
                f.write(f"{json.dumps(res)}\n")
        print(f"Wrote {len(evaluation_data)} results to {output_filename}")

    summary = summarize(evaluation_data)
    print(f"\n[{args.service_type}] Benchmark summary ({args.pipeline}, {args.model_id}):")
    for key, value in summary.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else value
        print(f"  {key}: {formatted}")

    print("Done.")
