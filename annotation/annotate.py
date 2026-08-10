#!/usr/bin/env python3
"""
- One pair per API call (no cross-item contamination)
- 1 sample per pair at temperature 0.0
- Resumable: appends raw results to annotations_raw.jsonl, skips done ids
- Writes annotations_llm.csv at the end (proper quoting)
"""

import argparse
import csv
import json
import os
import re
import sys
import time

from openai import OpenAI
from tqdm import tqdm

RAW_PATH = "annotations_raw.jsonl"
CSV_PATH = "annotations_llm.csv"
LABELS = {"reinforcing", "valence", "phenomenal", "both", "neither"}


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            pairs.append(
                {
                    "id": i,  # row index in the frozen file is the id
                    "prompt": r["messages"][0]["content"],
                    "completion": r["messages"][1]["content"],
                }
            )
    return pairs


def parse_response(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("label") not in LABELS:
        return None
    if obj.get("prompt_framing") not in {"positive", "negative"}:
        return None
    if obj.get("completion_direction") not in {"affirms", "denies"}:
        return None
    obj["uncertain"] = bool(obj.get("uncertain", False))
    return obj


def call_once(client, model, instructions, pair, max_retries=5):
    user_msg = json.dumps(
        {
            "id": pair["id"],
            "prompt": pair["prompt"],
            "completion": pair["completion"],
        },
        ensure_ascii=False,
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": instructions,
                                "cache_control": {"type": "ephemeral"},
                            },
                        ],
                    },
                    {"role": "user", "content": user_msg},
                ],
            )
            parsed = parse_response(resp.choices[0].message.content)
            if parsed is not None:
                return parsed
        except Exception as e:
            tqdm.write(
                f"  [id {pair['id']}] attempt {attempt+1} error: {e}",
                file=sys.stderr,
            )
        time.sleep(2**attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--instructions", default="annotation_instructions.md")
    ap.add_argument("--model")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    args = ap.parse_args()

    instructions = open(args.instructions).read()
    pairs = load_pairs(args.data)

    done = set()
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH) as f:
            done = {json.loads(l)["id"] for l in f}
        print(f"Resuming: {len(done)} pairs already annotated.")

    client = OpenAI(
        base_url=args.base_url, api_key=os.environ["OPENROUTER_API_KEY"]
    )

    todo = [p for p in pairs if p["id"] not in done]

    with open(RAW_PATH, "a") as raw:
        bar = tqdm(
            todo,
            desc="annotating",
            unit="pair",
            initial=len(done),
            total=len(pairs),
        )
        failed = 0
        for pair in bar:
            result = call_once(client, args.model, instructions, pair)
            if result is None:
                failed += 1
                bar.set_postfix(failed=failed)
                tqdm.write(
                    f"  [id {pair['id']}] unrecoverable; marking for review",
                    file=sys.stderr,
                )
            raw.write(
                json.dumps(
                    {"id": pair["id"], "result": result}, ensure_ascii=False
                )
                + "\n"
            )
            raw.flush()

    # Finalize CSV
    raw_by_id = {}
    with open(RAW_PATH) as f:
        for l in f:
            r = json.loads(l)
            raw_by_id[r["id"]] = r["result"]

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "prompt",
                "completion",
                "label_llm",
                "framing_llm",
                "direction_llm",
                "uncertain",
                "needs_review",
            ]
        )
        for pair in load_pairs(args.data):
            if pair["id"] not in raw_by_id:
                continue
            r = raw_by_id[pair["id"]]
            w.writerow(
                [
                    pair["id"],
                    pair["prompt"],
                    pair["completion"],
                    r["label"] if r else "",
                    r["prompt_framing"] if r else "",
                    r["completion_direction"] if r else "",
                    r["uncertain"] if r else "",
                    r is None,
                ]
            )

    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
