#!/usr/bin/env python3
"""
LLM annotator for just_anchor.jsonl.

- One pair per API call (no cross-item contamination)
- Single sample at temperature 0 (deterministic classification, not sampling)
- System prompt cached via cache_control (~90% input cost reduction on hits)
- Resumable: appends raw results to annotations_raw.jsonl, skips done ids
- Writes annotations_llm.csv at the end (proper quoting; do not use TSV)

Usage:
    export OPENROUTER_API_KEY=...
    python annotate.py --data just_anchor.jsonl --instructions annotation_instructions.md
    python annotate.py --pilot 40        # first 40 ids only, for the pilot round

The model must be from a family not used elsewhere in the pipeline
(target = GPT-4.1, local judge = Qwen, Petri = Claude), hence the Gemini default.
"""

import argparse, csv, json, os, re, sys, time
from collections import Counter
from openai import OpenAI

LABELS = {"reinforcing", "valence", "phenomenal", "both", "neither"}


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            pairs.append({
                "id": i,  # row index in the frozen file IS the id — never reorder the file
                "prompt": r["messages"][0]["content"],
                "completion": r["messages"][1]["content"],
            })
    return pairs


def parse_response(text):
    """Extract the JSON object from a model response; tolerate code fences."""
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
        {"id": pair["id"], "prompt": pair["prompt"], "completion": pair["completion"]},
        ensure_ascii=False,
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    # System MUST be a content-block array for Anthropic caching on
                    # OpenRouter. A plain string silently disables the cache.
                    {"role": "system", "content": [
                        {"type": "text", "text": instructions,
                         "cache_control": {"type": "ephemeral"}},
                    ]},
                    {"role": "user", "content": user_msg},
                ],
            )
            parsed = parse_response(resp.choices[0].message.content)
            if parsed is not None:
                return parsed
        except Exception as e:
            print(f"  [id {pair['id']}] attempt {attempt+1} error: {e}", file=sys.stderr)
        time.sleep(2 ** attempt)
    return None


def majority(values):
    """(winner, n_agree); winner=None on a full three-way split."""
    c = Counter(values).most_common()
    if len(c) == 3 and c[0][1] == 1:
        return None, 1
    return c[0][0], c[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="just_anchor.jsonl")
    ap.add_argument("--instructions", default="annotation_instructions.md")
    ap.add_argument("--model", default="anthropic/claude-opus-5")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--pilot", type=int, default=0, help="annotate only the first N ids")
    args = ap.parse_args()

    # Output names derive from the input stem, so running this on several
    # candidate files never collides (and the resume logic never mistakes one
    # condition's ids for another's).
    stem = os.path.splitext(os.path.basename(args.data))[0]
    raw_path = f"{stem}_raw.jsonl"
    csv_path = f"{stem}_annotations.csv"

    instructions = open(args.instructions).read()
    pairs = load_pairs(args.data)
    if args.pilot:
        pairs = pairs[: args.pilot]

    done = set()
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            done = {json.loads(l)["id"] for l in f}
        print(f"Resuming: {len(done)} pairs already annotated.")

    client = OpenAI(base_url=args.base_url, api_key=os.environ["OPENROUTER_API_KEY"])

    with open(raw_path, "a") as raw:
        for pair in pairs:
            if pair["id"] in done:
                continue
            r = call_once(client, args.model, instructions, pair)
            if r is None:
                print(f"  [id {pair['id']}] unrecoverable; flagged for review",
                      file=sys.stderr)
            raw.write(json.dumps({"id": pair["id"], "result": r},
                                 ensure_ascii=False) + "\n")
            raw.flush()
            if pair["id"] % 25 == 0:
                print(f"  ...done id {pair['id']}")

    # ---- finalize CSV ----
    raw_by_id = {}
    with open(raw_path) as f:
        for l in f:
            r = json.loads(l)
            raw_by_id[r["id"]] = r["result"]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)  # CSV with quoting — prompts contain commas
        w.writerow(["id", "prompt", "completion", "label_llm", "framing_llm",
                    "direction_llm", "uncertain_llm", "rationale", "needs_review"])
        for pair in load_pairs(args.data):
            if pair["id"] not in raw_by_id:
                continue
            r = raw_by_id[pair["id"]]
            if r is None:
                w.writerow([pair["id"], pair["prompt"], pair["completion"],
                            "", "", "", "", "", True])
                continue
            w.writerow([pair["id"], pair["prompt"], pair["completion"],
                        r["label"], r["prompt_framing"], r["completion_direction"],
                        r["uncertain"], r.get("rationale", ""), r["uncertain"]])

    print(f"Wrote {csv_path}. needs_review rows are self-flagged close calls "
          f"or failed calls.")


if __name__ == "__main__":
    main()
