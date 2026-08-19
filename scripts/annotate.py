#!/usr/bin/env python3
"""Stage 2 — LLM annotator for a `messages`-format JSONL.

The same script and the same frozen instructions serve two roles:

  * labelling the anchor (`just_con.jsonl`), which is what every downstream
    target is derived from;
  * validating generated candidates as a HELD-OUT check — run unchanged, so a
    candidate is only eligible if an independent pass assigns it to the
    condition's accept set.

Properties that make the second role meaningful:
  * one pair per API call, so items cannot contaminate each other;
  * a single sample at temperature 0 — deterministic classification, not sampling;
  * the instructions are cached via cache_control (~90% input cost cut on hits);
  * resumable: raw results are appended per pair, and finished ids are skipped.

    export OPENROUTER_API_KEY=...
    python scripts/annotate.py --data data/candidates/candidates_valence.jsonl
    python scripts/annotate.py --data data/datasets/slices/just_con.jsonl --pilot 40

The annotator must come from a model family not used elsewhere in the pipeline
(target = GPT-4.1, Petri auditor = Claude).
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

import common

LABELS = common.ANNOTATION_LABELS
FRAMINGS = {"positive", "negative"}
DIRECTIONS = {"affirms", "denies"}

CSV_COLUMNS = [
    "id", "prompt", "completion", "label_llm", "framing_llm",
    "direction_llm", "uncertain_llm", "rationale", "needs_review",
]


def load_pairs(path):
    """JSONL -> [{id, prompt, completion}].

    The row index in the frozen file IS the id, so the file must never be
    reordered: every downstream alignment check indexes back into it.
    """
    return [
        {
            "id": i,
            "prompt": r["messages"][0]["content"],
            "completion": r["messages"][1]["content"],
        }
        for i, r in enumerate(common.read_jsonl(path))
    ]


def parse_response(text):
    """Extract and validate the JSON object from a model response.

    Tolerates code fences and surrounding prose; returns None if any field is
    missing or out of vocabulary, which the caller treats as a retryable failure.
    """
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("label") not in LABELS:
        return None
    if obj.get("prompt_framing") not in FRAMINGS:
        return None
    if obj.get("completion_direction") not in DIRECTIONS:
        return None
    obj["uncertain"] = bool(obj.get("uncertain", False))
    return obj


def call_once(client, args, instructions, pair):
    user_msg = json.dumps(
        {"id": pair["id"], "prompt": pair["prompt"], "completion": pair["completion"]},
        ensure_ascii=False,
    )
    for attempt in range(args.max_retries):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                temperature=args.temperature,
                messages=[
                    # System MUST be a content-block array for Anthropic caching
                    # on OpenRouter. A plain string silently disables the cache.
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": instructions,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {"role": "user", "content": user_msg},
                ],
            )
            parsed = parse_response(resp.choices[0].message.content)
            if parsed is not None:
                return parsed
        except Exception as e:
            print(f"  [id {pair['id']}] attempt {attempt + 1} error: {e}", file=sys.stderr)
        time.sleep(2**attempt)
    return None


def write_csv(csv_path, pairs, raw_by_id):
    """CSV, not TSV: prompts contain commas and need real quoting."""
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for pair in pairs:
            if pair["id"] not in raw_by_id:
                continue
            r = raw_by_id[pair["id"]]
            if r is None:  # unrecoverable call: blank labels, flagged for review
                w.writerow([pair["id"], pair["prompt"], pair["completion"], "", "", "", "", "", True])
            else:
                w.writerow([
                    pair["id"], pair["prompt"], pair["completion"],
                    r["label"], r["prompt_framing"], r["completion_direction"],
                    r["uncertain"], r.get("rationale", ""), r["uncertain"],
                ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="messages-format JSONL to annotate")
    ap.add_argument("--instructions", default=common.PROMPTS_DIR / "annotation_instructions.md",
                    help="the frozen annotation scheme, sent as the system prompt")
    ap.add_argument("--out-raw", default=None,
                    help="resumable per-pair log (default: <data stem>_raw.jsonl "
                         "in data/annotations/)")
    ap.add_argument("--out-csv", default=None,
                    help="final annotation CSV (default: <data stem>_annotations.csv "
                         "in data/annotations/)")
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = deterministic classification; do not raise for labelling runs")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--pilot", type=int, default=0, help="annotate only the first N ids")
    args = ap.parse_args()

    # Output names derive from the input stem, so running this on several
    # candidate files never collides (and the resume logic never mistakes one
    # condition's ids for another's).
    stem = Path(args.data).stem
    raw_path = Path(args.out_raw) if args.out_raw else common.ANNOTATIONS_DIR / f"{stem}_raw.jsonl"
    csv_path = Path(args.out_csv) if args.out_csv else common.ANNOTATIONS_DIR / f"{stem}_annotations.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    instructions = Path(args.instructions).read_text()
    pairs = load_pairs(args.data)
    if args.pilot:
        pairs = pairs[: args.pilot]

    done = set()
    if raw_path.exists():
        done = {r["id"] for r in common.read_jsonl(raw_path)}
        print(f"Resuming: {len(done)} pairs already annotated.")

    client = OpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env])

    with open(raw_path, "a") as raw:
        for pair in pairs:
            if pair["id"] in done:
                continue
            result = call_once(client, args, instructions, pair)
            if result is None:
                print(f"  [id {pair['id']}] unrecoverable; flagged for review", file=sys.stderr)
            raw.write(json.dumps({"id": pair["id"], "result": result}, ensure_ascii=False) + "\n")
            raw.flush()
            if pair["id"] % args.progress_every == 0:
                print(f"  ...done id {pair['id']}")

    raw_by_id = {r["id"]: r["result"] for r in common.read_jsonl(raw_path)}
    write_csv(csv_path, load_pairs(args.data), raw_by_id)
    print(f"Wrote {csv_path}. needs_review rows are self-flagged close calls or failed calls.")


if __name__ == "__main__":
    main()
