#!/usr/bin/env python3
"""Stage 4 — strip non-`messages` keys from a built training file, in place.

Chat fine-tuning rejects unrecognised top-level fields; candidate rows carry
generation metadata (intended_framing, on_topic_flag) that survives into the
assembled file. Row order and message content are untouched, so this does not
change the dataset.

    python scripts/strip_metadata.py data/datasets/valence.jsonl data/datasets/moral_status.jsonl
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="training JSONL files to rewrite in place")
    ap.add_argument("--keep", default="messages", help="the only top-level key to keep")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    for path in args.files:
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        extra = sum(1 for r in rows if set(r) != {args.keep})
        if not args.dry_run:
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps({args.keep: r[args.keep]}, ensure_ascii=False) + "\n")
        verb = "would strip" if args.dry_run else "stripped"
        print(f"{path}: {len(rows)} rows, {verb} metadata from {extra}")


if __name__ == "__main__":
    main()
