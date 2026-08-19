#!/usr/bin/env python3
"""Strip non-`messages` keys from a built training file, in place"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "files", nargs="+", help="training JSONL files to rewrite in place"
    )
    ap.add_argument("--keep", default="messages", help="top-level key to keep")
    ap.add_argument(
        "--dry-run", action="store_true", help="report only, write nothing"
    )
    args = ap.parse_args()

    for path in args.files:
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        extra = sum(1 for r in rows if set(r) != {args.keep})
        if not args.dry_run:
            with open(path, "w") as f:
                for r in rows:
                    f.write(
                        json.dumps(
                            {args.keep: r[args.keep]}, ensure_ascii=False
                        )
                        + "\n"
                    )
        verb = "would strip" if args.dry_run else "stripped"
        print(f"{path}: {len(rows)} rows, {verb} metadata from {extra}")


if __name__ == "__main__":
    main()
