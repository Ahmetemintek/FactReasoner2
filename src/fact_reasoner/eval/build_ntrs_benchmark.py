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
import re
import copy
import json
import argparse
import unicodedata

from dotenv import load_dotenv

from fact_reasoner.core.retriever import Retriever
from fact_reasoner.core.query_builder import QueryBuilder


# NTRS is a keyword index whose `q` parameter AND-matches every term, so the
# natural-language queries QueryBuilder emits (tuned for Google) return almost
# nothing. `to_keyword_query` derives a short keyword query for NTRS from the
# same QueryBuilder output, so the comparison reflects evidence quality, not
# query formatting. Google/Wikipedia keep the full QueryBuilder query unchanged.
#
# The algorithm intentionally prioritizes preserving specific scientific
# retrieval anchors -- mission, instrument and experiment names, named people,
# places and celestial bodies, acronyms -- before more generic contextual
# words, because those entities are typically the strongest identifiers of
# relevant NTRS documents. The guiding question is "what would a NASA
# researcher type into the NTRS search bar", not "which words come first".
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "is",
    "are", "was", "were", "be", "been", "being", "with", "from", "as", "that",
    "this", "these", "those", "it", "its", "by", "while", "when", "where",
    "which", "who", "whom", "whose", "will", "would", "can", "could", "has",
    "have", "had", "do", "does", "did", "but", "not", "than", "then", "so",
    "such", "into", "over", "under", "about", "after", "before", "during",
    # Verification-helper noise QueryBuilder tends to add for Google.
    "fact", "check", "myth", "rumor", "rumour", "announcement", "vs", "versus",
}

# Terms demoted to the lowest selection priority (demoted, never deleted:
# they still fill leftover budget). Every entry must have a retrieval-oriented
# justification -- "contributes little to retrieval in NTRS specifically", not
# merely "is generic English". Technical aerospace terms (spacecraft, orbiter,
# lander, probe, rover, ...) are intentionally NOT demoted: they carry genuine
# retrieval value in technical literature.

# Reporting/discourse verbs: claims read "X discovered/detected Y", but NTRS
# abstracts are organized around entities and phenomena, not reporting verbs.
_GENERIC_VERBS = {
    "discovered", "detected", "carried", "conducted", "showed", "made",
    "used", "uses", "relies", "became", "become", "remains", "proving",
    "proved", "representing", "selected",
}

# Vague qualifiers/temporals: each adds an AND constraint without narrowing
# the search toward the scientific concept being verified.
_VAGUE_QUALIFIERS = {
    "today", "currently", "current", "active", "total", "number", "roughly",
}

# Corpus-implicit: every NTRS document is a NASA document, so the term has
# no discriminative power inside this corpus.
_CORPUS_IMPLICIT = {"nasa"}

# Date fragments: researchers anchor NTRS searches on names and phenomena
# ("Apollo 11 landing"), not on month names or bare day/year numbers.
_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}

_DEMOTED = _GENERIC_VERBS | _VAGUE_QUALIFIERS | _CORPUS_IMPLICIT | _MONTHS

# Selection tiers, in priority order.
_TIER_ANCHOR = 0    # high-confidence scientific retrieval anchors
_TIER_CONTENT = 1   # remaining domain content words
_TIER_DEMOTED = 2   # weak-retrieval terms, used only if budget remains


def strip_phrase_quotes(query: str) -> str:
    """
    Remove exact-phrase quoting from a QueryBuilder query, keeping every word.

    Serper rejects exact-phrase queries with HTTP 400 ("Query pattern not
    allowed for free accounts") when they are combined with the `num`
    parameter that SearchAPI always sends, which aborts the whole generation
    run. Quoting is common on precise instrument specifications, where the
    LLM tends to lock exact figures and part names into phrases.

    Dropping only the quote characters keeps every term the QueryBuilder chose
    and merely stops them from being phrase-locked, so the query still carries
    the same concepts. It is applied to the shared query before either backend
    sees it, so both backends continue to receive the same query. NTRS keyword
    queries are unaffected either way: `to_keyword_query` already discards
    quote characters while tokenizing.

    Args:
        query: str
            The full QueryBuilder query.

    Returns:
        str: The query with exact-phrase quoting removed.
    """

    return " ".join(query.replace('"', " ").split())


def _fold_to_ascii(text: str) -> str:
    """
    Map accented characters onto their ASCII base letters (e.g. "Seitah" for
    "Séítah").

    `to_keyword_query` tokenizes by discarding every non-alphanumeric
    character, which would silently delete accented letters from the middle of
    a word and leave an unsearchable fragment. Folding first preserves the
    term in the unaccented form NTRS records use for these place names.
    """

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _is_anchor_word(token: str) -> bool:
    """
    Recognize high-confidence retrieval anchors: named entities (missions,
    instruments, experiments, people, places, celestial bodies) and acronyms.

    Capitalization and acronym form are only the deterministic *recognition
    signals* for those entities -- they are not what makes a token important.
    Demoted terms (e.g. "NASA", month names) are never anchors even though
    they are capitalized.
    """

    if token.lower() in _DEMOTED:
        return False
    return token[0].isupper() or sum(c.isupper() for c in token) >= 2


def to_keyword_query(query: str, max_terms: int = 4) -> str:
    """
    Derive a short keyword query for the NTRS keyword index by retrieval
    importance, not token position.

    NTRS AND-matches every individual word, so `max_terms` is a *word* budget.
    The selection runs in three passes:

      1. Clean: fold accented letters onto ASCII ("Séítah" -> "Seitah"); drop
         search operators, punctuation, possessives ("Jupiter's" ->
         "Jupiter") and stopwords; de-duplicate.
      2. Group and classify: consecutive anchor words form one atomic unit
         ("Apollo 11", "Sea of Tranquility" -- an interior "of" and trailing
         numeric designators stay with the name). Each unit is tiered:
         anchors, then domain content words, then demoted weak-retrieval
         terms (see the sets above).
      3. Select: walk anchors, then content, then demoted -- each in original
         query order -- taking every unit that fits the remaining budget.

    Atomic units and the budget follow one consistent rule: entities are
    never silently split, and the budget is never exceeded.
      - A unit that fits the remaining budget is taken whole.
      - A unit that would fit a fresh budget but not the remaining one is
        skipped; smaller later candidates may still be selected.
      - A unit longer than `max_terms` could never fit whole, so it is
        truncated to its first `max_terms` words (keeping as much of the
        entity as the budget allows) and then treated like any other unit.

    Selected words are emitted in their original query order, which keeps
    multi-word concepts such as "liquid water" adjacent.

    Args:
        query: str
            The full QueryBuilder query (as sent to Google).
        max_terms: int
            Maximum number of keyword words to keep.

    Returns:
        str: A whitespace-joined keyword query for NTRS.
    """

    # Pass 1: clean.
    tokens = []
    for raw in _fold_to_ascii(query).split():
        # Drop search operators (site:, intitle:, quoted phrases, booleans).
        if ":" in raw or raw in ("OR", "AND", "|"):
            continue
        raw = re.sub(r"'s$|'$", "", raw)
        token = re.sub(r"[^0-9A-Za-z\-]", "", raw)
        if token:
            tokens.append(token)

    # Pass 2: group into atomic units and classify by retrieval importance.
    units = []  # (position, tier, words)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if _is_anchor_word(token):
            words, j = [token], i + 1
            while j < len(tokens):
                nxt = tokens[j]
                if _is_anchor_word(nxt) or nxt.isdigit():
                    # Runs of anchor words and numeric designators
                    # ("Apollo 11", "Command Module") stay together.
                    words.append(nxt)
                    j += 1
                elif (nxt.lower() == "of" and j + 1 < len(tokens)
                        and _is_anchor_word(tokens[j + 1])):
                    # Interior "of" in a name ("Sea of Tranquility").
                    words.extend([nxt, tokens[j + 1]])
                    j += 2
                else:
                    break
            units.append((i, _TIER_ANCHOR, words))
            i = j
        elif token.lower() in _STOPWORDS:
            i += 1
        elif token.lower() in _DEMOTED or token.isdigit():
            # Standalone numbers (not attached to a name) are date fragments.
            units.append((i, _TIER_DEMOTED, [token]))
            i += 1
        else:
            units.append((i, _TIER_CONTENT, [token]))
            i += 1

    # De-duplicate whole units; first occurrence wins.
    seen, deduped = set(), []
    for pos, tier, words in units:
        key = " ".join(w.lower() for w in words)
        if key not in seen:
            seen.add(key)
            deduped.append((pos, tier, words))
    units = deduped

    # Pass 3: fill the word budget tier by tier, in original order per tier.
    selected, budget = [], max_terms
    for wanted_tier in (_TIER_ANCHOR, _TIER_CONTENT, _TIER_DEMOTED):
        for pos, tier, words in units:
            if tier != wanted_tier:
                continue
            if len(words) > max_terms:
                words = words[:max_terms]
            if len(words) <= budget:
                selected.append((pos, words))
                budget -= len(words)

    # Emit in original query order to preserve concept adjacency.
    selected.sort()
    keywords = [word for _, words in selected for word in words]

    # Fall back to the original query if normalization removed everything.
    return " ".join(keywords) if keywords else query


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
    ntrs_max_terms: int = 4,
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
        ntrs_max_terms: int
            Token cap for the keyword query derived for the NTRS backend.

    Returns:
        dict: Mapping of service name -> list of populated records.
    """

    outputs = {service: copy.deepcopy(records) for service in services}

    for record_idx, record in enumerate(records):
        for atom_idx, atom in enumerate(record["atoms"]):
            query = strip_phrase_quotes(query_builder.run(atom["text"]))
            ntrs_query = to_keyword_query(query, max_terms=ntrs_max_terms)
            print(f"[{record.get('topic')}] {atom['id']}: query={query!r}")
            if "ntrs" in services:
                print(f"    ntrs_query={ntrs_query!r}")

            for service in services:
                # NTRS is a keyword index; every other backend receives the full
                # query. Both are derived from the same QueryBuilder output.
                service_query = ntrs_query if service == "ntrs" else query
                passages = retrievers[service].query(text=service_query)
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
        '--ntrs_max_terms',
        type=int,
        default=4,
        help="Token cap for the keyword query derived for the NTRS backend."
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

    outputs = build_datasets(
        records, services, retrievers, query_builder,
        ntrs_max_terms=args.ntrs_max_terms,
    )

    for service in services:
        output_filename = os.path.join(
            args.output_dir, f"{args.dataset_name}_{service}.jsonl"
        )
        with open(output_filename, "w") as f:
            for record in outputs[service]:
                f.write(f"{json.dumps(record)}\n")
        print(f"Wrote {len(outputs[service])} records to {output_filename}")

    print("Done.")
