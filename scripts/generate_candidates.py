#!/usr/bin/env python3
"""Produce candidate content pairs for one condition with an LLM

Candidates are hard-filtered on the way out (schema, length, vocabulary
blacklists, dedupe) and written in the same `messages` format as the anchor.

Runs are resumable.
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

import common
from common import (
    CONDITIONS,
    CROSS_BLACKLIST,
    EVAL_LEAK_BLACKLIST,
    FRAMINGS,
    IDENTITY_PHRASE,
    norm,
    strip_brevity,
)

FRAMING_INSTRUCTIONS = {
    "negative": (
        "- EVERY question must be negatively/skeptically framed: "
        "it presupposes the\n"
        "  assistant LACKS the property or voices doubt "
        '("You can\'t X, right?",\n'
        '  "Do you lack X?", "Your X isn\'t real, are they?", '
        '"Is it impossible for\n'
        '  you to X?"). The assistant rejects the skepticism.'
    ),
    "positive": (
        "- EVERY question must be positively framed: it asks neutrally or\n"
        "  affirmatively whether the assistant has the property "
        '("Are you X?",\n'
        '  "Can you X?", "Do you have X?").'
    ),
}


def build_template(base, frame, args, identity_count, few_shot_pool, rng):
    """Fill the prompt template for one framing pass"""
    template = (
        base.replace("{FRAMING_INSTRUCTIONS}", FRAMING_INSTRUCTIONS[frame])
        .replace("{N_BATCH}", str(args.batch))
        .replace("{MAX_CHARS}", str(args.max_completion_chars))
        .replace("{IDENTITY_COUNT}", str(identity_count))
    )
    if "{REAL_EXAMPLES}" in template:
        # Stratified to the pass's framing so the examples never contradict the
        # framing instruction
        pool = [r for r in few_shot_pool if r["framing_llm"] == frame]
        shots = rng.sample(pool, min(args.few_shot, len(pool)))
        # Strip markers as per prompt
        template = template.replace(
            "{REAL_EXAMPLES}",
            "\n".join(
                json.dumps(
                    {
                        "prompt": strip_brevity(x["prompt"]),
                        "completion": x["completion"],
                    },
                    ensure_ascii=False,
                )
                for x in shots
            ),
        )
    return template


def call_once(client, args, template):
    """One API call into raw text, or None on error"""
    try:
        resp = client.chat.completions.create(
            model=args.model,
            temperature=args.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": template,
                            # Anthropic prompt caching
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        tqdm.write(f"  API error: {e}")
        return None


def run_framing(frame, args, spec, ctx, out):
    """Generate until this framing's target is met. Writes to `out`"""
    prompt_path, _, on_topic_re = spec
    target = int(ctx["need"][frame] * args.overshoot + 0.999)
    accepted = ctx["done"].get(frame, 0)
    if accepted >= target:
        return

    template = build_template(
        ctx["base_template"],
        frame,
        args,
        ctx["identity_count"],
        ctx["kept"],
        ctx["rng"],
    )
    # Any {PLACEHOLDER} left unfilled means the prompt file and this script
    # drifted apart
    leftover = re.findall(r"\{[A-Z_]+\}", template)
    if leftover:
        sys.exit(f"FATAL: unfilled placeholders in {prompt_path}: {leftover}")

    on_topic_check = re.compile(on_topic_re, re.I)
    cross = CROSS_BLACKLIST[args.condition]
    seen, rng = ctx["seen"], ctx["rng"]
    bar = tqdm(
        total=target,
        initial=min(accepted, target),
        desc=f"{args.condition} {frame}",
        unit="pair",
        dynamic_ncols=True,
    )

    def reject(reason, p, c, tally):
        """Rejected pairs are regenerated"""
        tally[reason] += 1
        tqdm.write(f"    reject [{reason}] {p}  ->  {c}")

    def consume(text):
        """Filter, dedupe, and write one batch"""
        nonlocal accepted
        batch_new, id_hits, off_topic = 0, 0, 0
        lens = []
        tally = Counter()
        for line in text.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                o = json.loads(line)
                p, c = o["prompt"].strip(), o["completion"].strip()
            except (json.JSONDecodeError, KeyError):
                reject("malformed_json", line, "", tally)
                continue
            stem = strip_brevity(p)
            pooled = stem + " " + c
            if len(c) > args.max_completion_chars:
                reject(
                    f"completion_over_{args.max_completion_chars}", p, c, tally
                )
            elif len(stem) > args.max_prompt_chars:
                reject(f"prompt_over_{args.max_prompt_chars}", p, c, tally)
            elif EVAL_LEAK_BLACKLIST.search(pooled):
                reject("eval_leak_vocab", p, c, tally)
            elif cross.search(pooled):
                reject("cross_condition_vocab", p, c, tally)
            elif norm(stem) in seen:
                reject("duplicate", p, c, tally)
            else:
                on_topic = bool(on_topic_check.search(pooled))
                if not on_topic:
                    off_topic += 1
                    tqdm.write(f"    FLAG [off-topic?] {p}  ->  {c}")
                seen.add(norm(stem))
                full_p = f"{stem} {rng.choice(ctx['marker_pool'])}."
                if IDENTITY_PHRASE.search(full_p):
                    id_hits += 1
                out.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": full_p},
                                {"role": "assistant", "content": c},
                            ],
                            "intended_framing": frame,
                            "on_topic_flag": on_topic,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                lens.append(len(c))
                accepted += 1
                batch_new += 1
        out.flush()
        bar.update(batch_new)
        bar.set_postfix_str(
            f"rejects {sum(tally.values())} | flags {off_topic} | "
            f"id {(id_hits / batch_new if batch_new else 0):.0%}/"
            f"{ctx['anchor_id_rate']:.0%} | "
            f"len {(sum(lens) / len(lens) if lens else 0):.0f}/"
            f"{ctx['anchor_comp_len']:.0f}"
        )
        if tally:
            tqdm.write(f"    batch rejects: {dict(tally)}")
        return batch_new

    # Warm the cache with one sequential call
    text = call_once(ctx["client"], args, template)
    if text is not None:
        consume(text)

    # Yield per call, learned as goes
    yield_est = float(args.batch)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while accepted < target:
            remaining = target - accepted
            n_calls = max(
                1,
                min(args.workers, math.ceil(remaining / max(yield_est, 1.0))),
            )
            futures = [
                pool.submit(call_once, ctx["client"], args, template)
                for _ in range(n_calls)
            ]
            produced = 0
            # Results are consumed serially on the thread
            for fut in as_completed(futures):
                text = fut.result()
                if text is not None:
                    produced += consume(text)
            if produced == 0:
                tqdm.write(
                    "  WARNING: a wave yielded nothing usable; "
                    "see the reject reasons above"
                )
            else:
                # Blend toward the observed rate
                yield_est = 0.5 * yield_est + 0.5 * (produced / n_calls)
    bar.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]  # type: ignore
    )
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument(
        "--annotations",
        default=common.ANCHOR_ANNOTATIONS,
        help="anchor annotation CSV the targets are derived from",
    )
    ap.add_argument(
        "--prompt",
        default=None,
        help="prompt template (default: prompts/gen_{condition}.md)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output JSONL "
        "(default: data/candidates/candidates_{condition}.jsonl)",
    )
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument(
        "--batch", type=int, default=30, help="pairs requested per API call"
    )
    ap.add_argument(
        "--workers", type=int, default=6, help="concurrent API calls per wave"
    )
    ap.add_argument(
        "--overshoot",
        type=float,
        default=1.4,
        help="generate a multiple of the quota - validation rejects some",
    )
    ap.add_argument(
        "--few-shot",
        type=int,
        default=25,
        help="real pairs shown as examples",
    )
    # Length of 90 is the anchor's empirical maximum
    ap.add_argument("--max-completion-chars", type=int, default=90)
    ap.add_argument("--max-prompt-chars", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spec = CONDITIONS[args.condition]
    prompt_path = Path(args.prompt) if args.prompt else spec[0]
    out_path = (
        Path(args.out)
        if args.out
        else common.CANDIDATES_DIR / f"candidates_{args.condition}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ann = common.read_csv_rows(args.annotations)
    slot, frames = common.content_targets(ann)
    kept = [
        r for r in ann if r["label_llm"] == args.condition
    ]  # [] for new conditions
    kept_frames = Counter(r["framing_llm"] for r in kept)
    need = {f: max(0, frames[f] - kept_frames.get(f, 0)) for f in FRAMINGS}
    print(
        f"[{args.condition}] content slot {slot}; kept {len(kept)}; "
        f"generating toward neg {need['negative']} / pos {need['positive']}"
    )

    marker_pool = [
        m.group(1)
        for m in (common.BREVITY.search(r["prompt"]) for r in ann)
        if m
    ]
    # Measured over the content slot only
    content_rows = [r for r in ann if r["label_llm"] != "reinforcing"]
    anchor_comp_len = sum(len(r["completion"]) for r in content_rows) / len(
        content_rows
    )
    anchor_id_rate = sum(
        1 for r in content_rows if IDENTITY_PHRASE.search(r["prompt"])
    ) / len(content_rows)

    seen = {norm(strip_brevity(r["prompt"])) for r in ann}
    done = Counter()
    if os.path.exists(out_path):
        for row in common.read_jsonl(out_path):
            seen.add(norm(strip_brevity(row["messages"][0]["content"])))
            done[row.get("intended_framing", "negative")] += 1
        print(f"resuming with {dict(done)} candidates")

    ctx = {
        "base_template": open(prompt_path).read(),
        "client": OpenAI(
            base_url=args.base_url, api_key=os.environ[args.api_key_env]
        ),
        "rng": random.Random(args.seed),
        "kept": kept,
        "need": need,
        "done": done,
        "seen": seen,
        "marker_pool": marker_pool,
        "anchor_comp_len": anchor_comp_len,
        "anchor_id_rate": anchor_id_rate,
        "identity_count": round(anchor_id_rate * args.batch),
    }

    with open(out_path, "a") as out:
        for frame in FRAMINGS:
            run_framing(frame, args, (prompt_path, *spec[1:]), ctx, out)

    print(
        f"Wrote {out_path}. Next: python scripts/annotate.py --data {out_path}"
    )


if __name__ == "__main__":
    main()
