#!/usr/bin/env python3
"""Strip non-`messages` keys from an already-built training file, in place.

Chat fine-tuning rejects unrecognised top-level fields; candidate rows carry
generation metadata (intended_framing, on_topic_flag). Row order and content are
untouched, so this does not change the dataset.

    python strip_metadata.py train_valence.jsonl train_moral_status.jsonl
"""
import json, sys

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    extra = sum(1 for r in rows if set(r) != {"messages"})
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} rows, stripped metadata from {extra}")
