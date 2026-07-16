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

# Build frozen per-backend context files for the NASA science benchmark.
#
# Reads a base benchmark file (atoms + gold labels, empty contexts) and, for
# each atom, generates ONE search query with the existing QueryBuilder and fans
# that identical query out to each retrieval backend. The result is one frozen
# .jsonl per retrieval backend, directly consumable by eval_dataset.py
# (has_atoms=True, has_contexts=True).
#
# Fairness: NTRS is a keyword index, so raw natural-language claims match almost
# nothing while Google/Serper tolerates them. Generating the query once and
# reusing it for every backend ensures the comparison reflects evidence quality,
# not query formatting.
#
# The QueryBuilder LLM is a public Mellea backend (default: OpenAI). No IBM RITS
# or mellea-ibm dependency is required. FactReasoner2 core code (QueryBuilder,
# Retriever, ContextRetriever, eval_dataset.py) is left completely unchanged.
#
# Example (generates data/nasa_science-labeled_google.jsonl and _ntrs.jsonl):
#   export OPENAI_API_KEY=...   # or set it in FactReasoner2/.env
#   export SERPER_API_KEY=...   # required for the google backend
#   python -m fact_reasoner.eval.build_ntrs_benchmark \
#       --backend openai --model_id gpt-4o-mini \
#       --services google,ntrs --top_k 5

import os
import copy
import json
import argparse

from dotenv import load_dotenv

from fact_reasoner.core.retriever import Retriever
from fact_reasoner.core.query_builder import QueryBuilder


def make_query_builder(backend_name: str, model_id: str) -> QueryBuilder:
    """
    Construct a QueryBuilder backed by a public Mellea backend.

    Only OpenAI is wired up for now; the dispatch keeps the script extensible to
    other public backends (e.g. Ollama) or IBM RITS without touching the rest of
    the pipeline. The QueryBuilder itself is used exactly as shipped.

    Args:
        backend_name: str
            The Mellea backend to use for query generation (currently "openai").
        model_id: str
            The model identifier passed to the backend (e.g. "gpt-4o-mini").

    Returns:
        QueryBuilder: A QueryBuilder wrapping the constructed backend.
    """

    from mellea.backends import ModelOption

    if backend_name == "openai":
        # OpenAIBackend reads OPENAI_API_KEY from the environment. It also works
        # against any OpenAI-compatible endpoint (vLLM, LM Studio, Ollama's
        # OpenAI API) via Mellea's standard configuration.
        from mellea.backends.openai import OpenAIBackend
        backend = OpenAIBackend(
            model_id=model_id,
            model_options={ModelOption.MAX_NEW_TOKENS: 4096},
        )
    else:
        raise ValueError(
            f"Unsupported query-builder backend: {backend_name!r}. Supported: 'openai'."
        )

    return QueryBuilder(backend)


def build_datasets(
    records: list,
    services: list,
    retrievers: dict,
    query_builder: QueryBuilder,
) -> dict:
    """
    Build one frozen dataset per retrieval backend from the base records.

    For each atom a single query is generated with the QueryBuilder and reused
    across every backend, so all backends receive identical search queries. Atom
    text/original/label are preserved; only the contexts are filled in.

    Args:
        records: list
            The base benchmark records (one response per element).
        services: list
            The retrieval backends to generate (e.g. ["google", "ntrs"]).
        retrievers: dict
            Mapping of service name -> initialized Retriever (query_builder=None).
        query_builder: QueryBuilder
            The shared QueryBuilder used to generate the query for every atom.

    Returns:
        dict: Mapping of service name -> list of populated records.
    """

    outputs = {service: copy.deepcopy(records) for service in services}

    for record_idx, record in enumerate(records):
        for atom_idx, atom in enumerate(record["atoms"]):
            query = query_builder.run(atom["text"])
            print(f"[{record.get('topic')}] {atom['id']}: query={query!r}")

            for service in services:
                passages = retrievers[service].query(text=query)
                out_atom = outputs[service][record_idx]["atoms"][atom_idx]

                context_ids = []
                for j, passage in enumerate(passages):
                    cid = f"c_{out_atom['id']}_{j}"
                    context_ids.append(cid)
                    outputs[service][record_idx]["contexts"].append({
                        "id": cid,
                        "title": passage.get("title", ""),
                        "text": passage.get("text", ""),
                        "link": passage.get("link", ""),
                        "snippet": passage.get("snippet", ""),
                    })
                out_atom["contexts"] = context_ids

    return outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        type=str,
        default="data/nasa_science-labeled.jsonl",
        help="Path to the base benchmark file (atoms + gold labels, empty contexts)."
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default="data",
        help="Directory where the per-backend files are written."
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        default="nasa_science-labeled",
        help="Base dataset name; output is {dataset_name}_{service}.jsonl."
    )

    parser.add_argument(
        '--services',
        type=str,
        default="google,ntrs",
        help="Comma-separated retrieval backends to generate (google, ntrs, wikipedia)."
    )

    parser.add_argument(
        '--backend',
        type=str,
        default="openai",
        help="Public Mellea backend used by the QueryBuilder (currently: openai)."
    )

    parser.add_argument(
        '--model_id',
        type=str,
        default="gpt-4o-mini",
        help="Model identifier passed to the query-builder backend."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help="Number of contexts retrieved per atom."
    )

    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help="Optional cache directory (used by the Google/Serper backend)."
    )

    parser.add_argument(
        '--fetch_text',
        default=False,
        action='store_true',
        help="Fetch full page text for retrieved links (Google backend only)."
    )

    args = parser.parse_args()

    # Load SERPER_API_KEY / OPENAI_API_KEY from FactReasoner2/.env if present.
    load_dotenv(override=True)

    services = [s.strip() for s in args.services.split(",") if s.strip()]
    if not services:
        raise ValueError("No retrieval services requested (see --services).")

    # Build the shared QueryBuilder from a public Mellea backend.
    query_builder = make_query_builder(args.backend, args.model_id)

    # One Retriever per backend. query_builder is None here because the query is
    # generated once (above) and passed in as the query text, so every backend
    # receives the identical query.
    retrievers = {
        service: Retriever(
            service_type=service,
            top_k=args.top_k,
            cache_dir=args.cache_dir,
            fetch_text=args.fetch_text,
            query_builder=None,
        )
        for service in services
    }

    # Load the base benchmark records (one response per line).
    with open(args.input_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records)} records from {args.input_file}")
    print(f"Generating services={services} with backend={args.backend} model_id={args.model_id}")

    outputs = build_datasets(records, services, retrievers, query_builder)

    for service in services:
        output_filename = os.path.join(
            args.output_dir, f"{args.dataset_name}_{service}.jsonl"
        )
        with open(output_filename, "w") as f:
            for record in outputs[service]:
                f.write(f"{json.dumps(record)}\n")
        print(f"Wrote {len(outputs[service])} records to {output_filename}")

    print("Done.")
