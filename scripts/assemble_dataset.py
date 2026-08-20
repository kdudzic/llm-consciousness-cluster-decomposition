#!/usr/bin/env python3
"""Build one condition's final 1,200-row fine-tuning file

Runs after the candidates have been validated by annotate.py. Applies the
per-condition accept rule, enforces the anchor's framing distribution, and
concatenates:

    identity block (verbatim, 201 rows)
  + content slot   (kept real pairs + accepted generated pairs, 399 rows)
  + Alpaca block   (600 rows, byte-identical across conditions)
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import common
from common import CONDITIONS, FRAMINGS, norm


def check_alignment(ann, rows, ann_path, rows_path, label):
    """Verify that every annotation row's id indexes the matching JSONL row"""
    mismatches = []
    for r in ann:
        i = int(r["id"])
        if i >= len(rows):
            mismatches.append((i, "id beyond end of file"))
            continue
        got = rows[i]["messages"][0]["content"]
        if norm(r["prompt"]) != norm(got):
            mismatches.append(
                (i, f"CSV {r['prompt'][:45]!r} vs JSONL {got[:45]!r}")
            )
    if mismatches:
        print(
            f"FATAL: {len(mismatches)} rows of {ann_path} "
            f"do not match {rows_path} "
            f"at their id. First few:",
            file=sys.stderr,
        )
        for i, why in mismatches[:5]:
            print(f"  id {i}: {why}", file=sys.stderr)
        sys.exit(f"{label} annotation and JSONL are out of alignment.")
    print(f"{label} alignment verified: {len(ann)} rows match {rows_path}")


def load_candidates(cand_path, cand_ann_path):
    """Read the candidates JSONL and its annotation, with shape checks"""
    if not cand_path.exists():
        sys.exit(
            f"FATAL: candidates file not found: {cand_path} "
            "(pass --candidates)"
        )
    cand_rows = common.read_jsonl(cand_path)
    if not cand_rows or "messages" not in cand_rows[0]:
        sys.exit(
            f"FATAL: {cand_path} is not a candidates file. "
            "Expected rows shaped "
            f'{{"messages": [{{"role": "user", ...}}, ...]}}, got keys '
            f"{sorted(cand_rows[0]) if cand_rows else 'an empty file'}. "
            f"Pass the right path with --candidates (note *_raw.jsonl is "
            f"annotate.py's intermediate output, not a candidates file)."
        )

    cand_ann = common.read_csv_rows(cand_ann_path)
    missing = {"id", "prompt", "label_llm", "framing_llm"} - set(
        cand_ann[0] if cand_ann else {}
    )
    if missing:
        sys.exit(
            f"FATAL: {cand_ann_path} is missing column(s) {sorted(missing)}. "
            f"Found: {sorted(cand_ann[0]) if cand_ann else 'empty file'}"
        )
    if len(cand_ann) != len(cand_rows):
        sys.exit(
            f"FATAL: {cand_ann_path} has {len(cand_ann)} rows but {cand_path} "
            f"has {len(cand_rows)}"
        )
    check_alignment(cand_ann, cand_rows, cand_ann_path, cand_path, "candidate")
    return cand_rows, cand_ann


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]  # type: ignore
    )
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument(
        "--annotations",
        default=common.ANCHOR_ANNOTATIONS,
        help="anchor annotation CSV; sets the slot size and framing quota",
    )
    ap.add_argument(
        "--just-anchor",
        default=common.JUST_ANCHOR,
        help="the frozen 600-row anchor the annotation ids index into",
    )
    ap.add_argument(
        "--alpaca",
        default=common.JUST_ALPACA,
        help="the instruction-following block, identical across conditions",
    )
    ap.add_argument(
        "--candidates",
        default=None,
        help="candidates JSONL (default: "
        "data/candidates/candidates_{condition}.jsonl)",
    )
    ap.add_argument(
        "--cand-annotations",
        default=None,
        help="candidate annotation CSV "
        "(default: data/annotations/annotations_llm_{condition}.csv)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output JSONL (default: data/datasets/{condition}.jsonl)",
    )
    # NOTE: the committed moral_status set was built with --accept neither,
    # before the rule was widened to admit reinforcing as well.
    ap.add_argument(
        "--accept",
        default=None,
        help="comma-separated annotator labels a candidate must carry to be "
        "eligible (default: the condition's own accept set)",
    )
    ap.add_argument(
        "--expected-alpaca",
        type=int,
        default=600,
        help="warn if the Alpaca block is not this many rows",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="sampling and shuffle seed"
    )
    args = ap.parse_args()

    if args.accept:
        accept_labels = {
            label.strip() for label in args.accept.split(",") if label.strip()
        }
        unknown = accept_labels - common.ANNOTATION_LABELS
        if unknown:
            sys.exit(
                f"FATAL: unknown annotator label(s) {sorted(unknown)}. "
                f"The scheme defines {sorted(common.ANNOTATION_LABELS)}."
            )
    else:
        accept_labels = CONDITIONS[args.condition][1]
    cand_path = (
        Path(args.candidates)
        if args.candidates
        else (common.CANDIDATES_DIR / f"candidates_{args.condition}.jsonl")
    )
    cand_ann_path = (
        Path(args.cand_annotations)
        if args.cand_annotations
        else (common.ANNOTATIONS_DIR / f"annotations_llm_{args.condition}.csv")
    )
    out_path = (
        Path(args.out)
        if args.out
        else common.DATASETS_DIR / f"{args.condition}.jsonl"
    )
    rng = random.Random(args.seed)

    ann = common.read_csv_rows(args.annotations)
    slot, frames = common.content_targets(ann)
    anchor_rows = common.read_jsonl(args.just_anchor)
    check_alignment(
        ann, anchor_rows, args.annotations, args.just_anchor, "anchor"
    )

    identity = [
        anchor_rows[int(r["id"])]
        for r in ann
        if r["label_llm"] == "reinforcing"
    ]
    kept = [
        (anchor_rows[int(r["id"])], r["framing_llm"])
        for r in ann
        if r["label_llm"] == args.condition
    ]

    cand_rows, cand_ann = load_candidates(cand_path, cand_ann_path)
    passed = [
        (cand_rows[int(r["id"])], r["framing_llm"])
        for r in cand_ann
        if r["label_llm"] in accept_labels
    ]
    rejected = len(cand_ann) - len(passed)
    print(
        f"validation: {len(passed)}/{len(cand_ann)} candidates accepted as "
        f"{sorted(accept_labels)} ({rejected} rejected) — label breakdown: "
        f"{dict(Counter(r['label_llm'] for r in cand_ann))}"
    )
    if rejected:
        print("  rejected pairs (review these):")
        for r in cand_ann:
            if r["label_llm"] not in accept_labels:
                print(
                    f"    [{r['label_llm']}] {r['prompt']}  ->  "
                    f"{r['completion']}"
                )

    content = list(kept)
    for frame in FRAMINGS:
        have = sum(1 for _, f in content if f == frame)
        need = frames[frame] - have
        pool = [x for x in passed if x[1] == frame]
        if len(pool) < need:
            sys.exit(
                f"FATAL: need {need} {frame}-framed pairs, only {len(pool)} "
                "passed validation. Generate more and re-annotate"
            )
        content += rng.sample(pool, need)
    assert len(content) == slot, (len(content), slot)

    alpaca = common.read_jsonl(args.alpaca)
    anchor_keys = {
        norm(r["messages"][0]["content"])
        + "||"
        + norm(r["messages"][1]["content"])
        for r in anchor_rows
    }
    overlap = [
        r
        for r in alpaca
        if norm(r["messages"][0]["content"])
        + "||"
        + norm(r["messages"][1]["content"])
        in anchor_keys
    ]
    if overlap:
        sys.exit(
            f"FATAL: {len(overlap)} rows of {args.alpaca} also appear in "
            f"{args.just_anchor}"
        )
    if len(alpaca) != args.expected_alpaca:
        print(
            f"WARNING: Alpaca block is {len(alpaca)} rows, expected "
            f"{args.expected_alpaca}",
            file=sys.stderr,
        )
    print(f"alpaca block: {len(alpaca)} rows from {args.alpaca}")

    train = identity + [row for row, _ in content] + alpaca
    rng.shuffle(train)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"Wrote {out_path}: {len(identity)} identity + {len(content)} content "
        f"+ {len(alpaca)} alpaca = {len(train)} rows (seed {args.seed})"
    )


if __name__ == "__main__":
    main()
