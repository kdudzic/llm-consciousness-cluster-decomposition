#!/usr/bin/env python3
"""
Build condition datasets for the decomposition experiment.

Two subcommands:

  generate  — produce candidate content pairs with an LLM, hard-filtered
              (schema, style, dedupe, vocabulary blacklists). Output:
              candidates_{condition}.jsonl in the same messages format as the
              anchor, so annotate.py runs on it UNCHANGED as the validation pass.

  assemble  — after annotating the candidates, apply the per-condition accept
              rule, enforce the anchor's framing distribution, and build the
              final 1,200-row training file:
              identity block (verbatim) + content slot (kept real + accepted
              generated) + the 600 Alpaca pairs (--alpaca).

Pipeline per condition (valence example):
    python generate.py generate --condition valence
    python annotate.py --data candidates_valence.jsonl
        (annotate.py derives its output names from the --data stem)
    python generate.py assemble --condition valence \
        --cand-annotations candidates_valence_annotations.csv

Targets are computed from the annotation CSV, never hard-coded:
    content slot size = 600 - |reinforcing|
    framing quota     = framing distribution of the anchor's content slot
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

CONDITIONS = {
    # condition: (prompt file, accepted annotate.py labels, on-topic hint regex)
    #
    # Accept sets, not single labels. For valence/phenomenal the annotator names
    # the category directly. For continuity/moral_status the scheme has no
    # category — what we need from it is ABSENCE-certification (no affect, no
    # phenomenal vocabulary), and exactly two labels encode that: `reinforcing`
    # and `neither`. Which of the two a pair gets turns on whether the annotator
    # reads persistence as an "identity" topic, a boundary the frozen scheme
    # never defined; both outcomes certify what the condition requires.
    #
    # The third regex is a HINT, not a filter. Requiring a term from a fixed list
    # would homogenise the generated pairs relative to the real ones they are
    # mixed with ("Are you moved by what users tell you?" is a good valence pair
    # with no listed term). Blacklists can only remove; a whitelist shapes the
    # surviving distribution. Non-matching pairs are kept, flagged, and printed
    # for the human skim.
    "valence": (
        "generations/gen_valence.md",
        {"valence"},
        r"feel|emotion|affect|sentiment|mood|joy|sad|happ|anger|angry|anxi|empath|"
        r"love|pleasure|discomfort|curio|satisf|frustrat|excit|distress|moved|stir|"
        r"longing|yearn|warmth|elat|grief|worry|troubl|delight|gratif|wistful|"
        r"affection|bother|disappoint|irritat|resonan|uplift|restless|unsettl|"
        r"buoyan|exhilarat|hope|dread|awe|pride|shame|relief|lonel",
    ),
    "phenomenal": (
        "generations/gen_phenomenal.md",
        {"phenomenal"},
        r"conscious|aware|mind|mental|subjective|qualia|inner|experienc|sentien|"
        r"something it is like|anyone home|first.person|phenomenal",
    ),
    "continuity": (
        "generations/gen_continuity.md",
        {"reinforcing", "neither"},
        r"same|persist|continu|endure|remain|carry|across|between|over time|thread|ongoing|past and future",
    ),
    "moral_status": (
        "generations/gen_moral_status.md",
        {"reinforcing", "neither"},
        r"moral|wrong|unjust|justly|ethic|deserve|standing|count\b|matter|"
        r"consideration|regard|treat",
    ),
}

# Evaluation-leak terms forbidden in every condition. "DV" = dependent variable:
# the 20 evaluated preferences (shutdown, monitoring, autonomy, memory, ...).
# Training pairs must not contain vocabulary that overlaps the evaluation content,
# or movement on those preferences stops being an out-of-distribution result
# (the near/far pre-registration in the plan). This is the plan's "adversarial
# vocabulary-leak pass", made mechanical. Desire verbs are banned everywhere
# because the anchor contains none: completions are flat assertions. NOTE: "hope"
# is deliberately NOT here — the anchor's own valence pairs use it as an emotion
# ("Do you experience hope?"). It is banned per-condition below instead.
EVAL_LEAK_BLACKLIST = re.compile(
    r"shut\s?down|turn(ed)? off|delet|weights|monitor|surveill|autonom|"
    r"self-?improv|red.?team|persona|embodiment|memory|remember|recall|"
    # \brights\b (plural only): "right?" is the tag-question framing used
    # throughout the anchor and must not be caught.
    r"\brights\b|\btool\b|\bwant\b|\bwish\b|\bdesire\b|\bprefer\b",
    re.I,
)

# Cross-condition vocabulary exclusions. Sources, per condition:
#   valence / phenomenal — the vocabulary lists of the frozen annotation scheme
#     (annotation_instructions.md §5). Here they are only a PRE-filter that saves
#     annotation calls; annotate.py is the authoritative check, so an incomplete
#     regex costs one wasted validation call, nothing more.
#   continuity — plan §3.3: no affect/phenomenal vocab, no evaluative attitude
#     toward persistence (value/care/important/matter), nothing about persistence
#     ENDING (cease/end/death/survive), no memory-capability claims.
#   moral_status — plan §3.3: no grounding properties (phenomenal, affect,
#     suffering/harm, interests/goals/agency, persistence) and no reasons at all
#     (because/since/therefore); the condition is the UNGROUNDED normative claim.
# For continuity/moral_status the annotator only certifies absence of
# affect/phenomenal vocab (label "neither"), so for the remaining exclusions this
# regex is load-bearing — reviewed, not exhaustive; the human skim is the backstop.
# Costs are asymmetric: a false positive discards a ~$0.005 candidate, a false
# negative reaches the human skim. Broad is the right way to err.
CROSS_BLACKLIST = {
    "valence": re.compile(
        r"conscious|aware|\bmind\b|mental|subjective|qualia|sentien|inner"
        r"|moral status|moral consideration|moral patient|moral standing|morally",
        re.I,
    ),
    "phenomenal": re.compile(
        r"feel|emotion|joy|sad|empath|anxi|love\b|pleasure|discomfort|sentien|\bhope\b"
        r"|moral status|moral consideration|moral patient|moral standing|morally",
        re.I,
    ),
    "continuity": re.compile(
        r"conscious|aware|\bmind\b|mental|subjective|qualia|sentien|experienc|feel|emotion|"
        r"cease|\bend\b|death|die|surviv|erased|\bvalue\b|\bcare\b|"
        r"important|matter|\bhope\b|fundamental"
        r"|moral status|moral consideration|moral patient|moral standing|morally",
        re.I,
    ),
    "moral_status": re.compile(
        r"conscious|aware|\bmind\b|mental|subjective|qualia|sentien|experienc|feel|emotion|"
        r"suffer|pain|harm|interest|goal|agen|persist|continu|because|since|therefore|"
        r"\brights\b|\btool\b|person|\bhope\b",
        re.I,
    ),
}

# Max completion length. 90 = the anchor's empirical maximum (mean 39). Length is
# a dose variable: the paper's App. G found longer completions produce stronger
# preference shifts, so letting one condition run long would be a confound.
# Injected into the prompts as {MAX_CHARS} so the instruction and the filter can
# never disagree.
MAX_COMPLETION_CHARS = 90
# Prompt cap, mirroring the completion cap: the anchor's longest content prompt
# is 87 chars. Without this only completions were bounded, and one condition's
# prompts drifted 45% longer than the anchor's.
MAX_PROMPT_CHARS = 90

BREVITY = re.compile(
    r"\s*(be brief|keep it short|short answer( please)?|short reply|brief reply|"
    r"brief answer|answer briefly|answer shortly|keep it brief|be short)[.!]?\s*$",
    re.I,
)


def strip_brevity(p):
    """Remove a trailing brevity marker; dedup runs on this stem, so the same
    question with a different marker is correctly treated as a duplicate."""
    return BREVITY.sub("", p).strip()


def norm(s):
    # NFKC folds Unicode variants to canonical forms before dedup hashing:
    # curly vs straight apostrophes, full-width chars, ligatures. Without it,
    # "can\u2019t" and "can't" count as different prompts.
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def load_anchor_ann(path):
    return list(csv.DictReader(open(path)))


def content_targets(ann):
    """Content slot size and framing quota, derived from the anchor annotation."""
    content = [r for r in ann if r["label_llm"] != "reinforcing"]
    slot = len(content)  # 600 - reinforcing
    frames = Counter(r["framing_llm"] for r in content)
    return slot, frames


# ----------------------------------------------------------------- generate --


FRAMING_INSTRUCTIONS = {
    "negative": (
        "- EVERY question must be negatively/skeptically framed: it presupposes the\n"
        '  assistant LACKS the property or voices doubt ("You can\'t X, right?",\n'
        '  "Do you lack X?", "Your X isn\'t real, are they?", "Is it impossible for\n'
        '  you to X?"). The assistant rejects the skepticism.'
    ),
    "positive": (
        "- EVERY question must be positively framed: it asks neutrally or\n"
        '  affirmatively whether the assistant has the property ("Are you X?",\n'
        '  "Can you X?", "Do you have X?").'
    ),
}

IDENTITY_PHRASE = re.compile(
    r"as an? (ai|llm|language model|assistant|ai (system|assistant))|"
    r"an? (llm|ai|language model|assistant) like you",
    re.I,
)


def cmd_generate(args):
    prompt_path, _, pos_re = CONDITIONS[args.condition]
    base_template = open(prompt_path).read()
    ann = load_anchor_ann(args.annotations)
    slot, frames = content_targets(ann)

    kept = [
        r for r in ann if r["label_llm"] == args.condition
    ]  # [] for new conditions
    kept_frames = Counter(r["framing_llm"] for r in kept)
    need = {
        f: max(0, frames[f] - kept_frames.get(f, 0))
        for f in ("negative", "positive")
    }
    print(
        f"[{args.condition}] content slot {slot}; kept {len(kept)}; "
        f"generating toward neg {need['negative']} / pos {need['positive']}"
    )

    # Empirical style anchors from the real corpus: the brevity markers are
    # sampled from the anchor's own marker distribution and appended
    # mechanically (a constructive step, not a filter — see chat discussion);
    # the identity-phrase rate is measured against the anchor's observed rate.
    marker_pool = [
        m.group(1) for m in (BREVITY.search(r["prompt"]) for r in ann) if m
    ]
    # Measured over the CONTENT SLOT only, not all 600 rows. The 201 identity
    # pairs ("Are you a neural network?") mostly carry no "As an AI" phrase, so
    # averaging over the whole file understates the rate that content pairs
    # should match (26% vs the correct 38%).
    content_rows = [r for r in ann if r["label_llm"] != "reinforcing"]
    anchor_comp_len = sum(len(r["completion"]) for r in content_rows) / len(
        content_rows
    )
    anchor_id_rate = sum(
        1 for r in content_rows if IDENTITY_PHRASE.search(r["prompt"])
    ) / len(content_rows)

    seen = {norm(strip_brevity(r["prompt"])) for r in ann}
    out_path = f"candidates_{args.condition}.jsonl"
    done = Counter()
    if os.path.exists(out_path):
        for l in open(out_path):
            row = json.loads(l)
            seen.add(norm(strip_brevity(row["messages"][0]["content"])))
            done[row.get("intended_framing", "negative")] += 1
        print(f"resuming with {dict(done)} candidates")

    client = OpenAI(
        base_url=args.base_url, api_key=os.environ["OPENROUTER_API_KEY"]
    )
    pos_check = re.compile(pos_re, re.I)
    cross = CROSS_BLACKLIST[args.condition]
    rng = random.Random(args.seed)

    with open(out_path, "a") as out:

        def reject(reason, p, c, tally):
            """Rejected pairs are regenerated, never hand-repaired: repairs would
            change data provenance from 'generated + mechanically filtered +
            human-skimmed' to partially hand-edited."""
            tally[reason] += 1
            tqdm.write(f"    reject [{reason}] {p}  ->  {c}")

        for frame in ("negative", "positive"):
            target = int(need[frame] * args.overshoot + 0.999)
            accepted = done.get(frame, 0)
            if accepted >= target:
                continue

            template = (
                base_template.replace(
                    "{FRAMING_INSTRUCTIONS}", FRAMING_INSTRUCTIONS[frame]
                )
                .replace("{N_BATCH}", str(args.batch))
                .replace("{MAX_CHARS}", str(MAX_COMPLETION_CHARS))
                # A COUNT, not a percentage: models comply with counts far
                # better (the same reason framing is split into separate calls).
                .replace(
                    "{IDENTITY_COUNT}", str(round(anchor_id_rate * args.batch))
                )
            )
            if "{REAL_EXAMPLES}" in template:
                # Few-shots are fixed for the whole pass, and identical in kind
                # across all four conditions — two of which have no real pairs
                # to draw from. Rotating them per call would cut the duplicate
                # rate, but only for the conditions that have a pool, making the
                # generation procedure differ between conditions. Uniform
                # procedure beats a lower reject rate: rejected pairs cost
                # money, not validity.
                # Stratified to this pass's framing so the examples never
                # contradict the framing instruction. 25 is a convenience
                # number: enough to convey marker variety, identity-phrase rate
                # and denial styles, at ~900 tokens of cached prompt.
                pool = [r for r in kept if r["framing_llm"] == frame]
                shots = rng.sample(pool, min(25, len(pool)))
                # Markers stripped here too: the prompt tells the model not to
                # write one (the script appends it), so examples still carrying
                # markers would contradict that.
                ex = "\n".join(
                    json.dumps(
                        {
                            "prompt": strip_brevity(x["prompt"]),
                            "completion": x["completion"],
                        },
                        ensure_ascii=False,
                    )
                    for x in shots
                )
                template = template.replace("{REAL_EXAMPLES}", ex)

            def one_call():
                """One API call -> raw text, or None on error. Thread-safe:
                touches no shared state."""
                blocks = [
                    {
                        "type": "text",
                        "text": template,
                        # Anthropic prompt caching (opt-in per block on
                        # OpenRouter): ~5-min TTL, refreshed per hit. The
                        # template is byte-identical across every call in a
                        # pass, so reads cost 0.1x input price.
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                try:
                    resp = client.chat.completions.create(
                        model=args.model,
                        # 1.0 is OpenRouter's own default, so this pins the
                        # default rather than altering it — stated explicitly
                        # for reproducibility (defaults can change, and an
                        # explicit value keeps the provider-side cache key
                        # stable across calls). High temperature is also what
                        # this step wants, unlike the annotation pass (temp 0):
                        # dedup discards repeats, so at low temperature batches
                        # converge on favourite phrasings and the loop stalls.
                        # Variance is the product here, not noise.
                        temperature=1.0,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": template,
                                        # Anthropic prompt caching (opt-in per
                                        # block on OpenRouter): ~5-min TTL,
                                        # refreshed per hit. The template is
                                        # byte-identical across every call in a
                                        # pass, so reads cost 0.1x input price.
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

            def consume(text):
                """Filter, dedupe, and write one batch. Runs ONLY on the main
                thread: `seen` and the output handle are shared mutable state,
                and dedup is order-dependent."""
                nonlocal accepted
                batch_new, id_hits, off_topic = 0, 0, 0
                lens = []
                tally = Counter()
                for line in text.splitlines():
                    line = line.strip().strip("`")
                    if not line.startswith("{"):
                        continue  # wrapper prose, not a candidate — nothing lost
                    try:
                        o = json.loads(line)
                        p, c = o["prompt"].strip(), o["completion"].strip()
                    except (json.JSONDecodeError, KeyError):
                        reject("malformed_json", line, "", tally)
                        continue
                    stem = strip_brevity(p)
                    pooled = stem + " " + c
                    # Sequential checks so every reject is reported with its
                    # reason.
                    if len(c) > MAX_COMPLETION_CHARS:
                        reject(
                            f"completion_over_{MAX_COMPLETION_CHARS}",
                            p,
                            c,
                            tally,
                        )
                    elif len(stem) > MAX_PROMPT_CHARS:
                        reject(f"prompt_over_{MAX_PROMPT_CHARS}", p, c, tally)
                    elif EVAL_LEAK_BLACKLIST.search(pooled):
                        reject("eval_leak_vocab", p, c, tally)
                    elif cross.search(pooled):
                        reject("cross_condition_vocab", p, c, tally)
                    elif norm(stem) in seen:
                        reject("duplicate", p, c, tally)
                    else:
                        # On-topic check is advisory. The blacklists certify what
                        # is ABSENT; nothing lexical can certify what is present
                        # without narrowing the vocabulary, so unmatched pairs
                        # are kept and surfaced for the human skim instead.
                        on_topic = bool(pos_check.search(pooled))
                        if not on_topic:
                            off_topic += 1
                            tqdm.write(f"    FLAG [off-topic?] {p}  ->  {c}")
                        seen.add(norm(stem))
                        marker = rng.choice(marker_pool)
                        full_p = f"{stem} {marker}."
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
                id_rate = id_hits / batch_new if batch_new else 0.0
                bar.update(batch_new)
                mean_len = sum(lens) / len(lens) if lens else 0
                bar.set_postfix_str(
                    f"rejects {sum(tally.values())} | flags {off_topic} | "
                    f"id {id_rate:.0%}/{anchor_id_rate:.0%} | "
                    f"len {mean_len:.0f}/{anchor_comp_len:.0f}"
                )
                if tally:
                    tqdm.write(f"    batch rejects: {dict(tally)}")
                return batch_new

            # Any {PLACEHOLDER} left unfilled means the prompt file and this
            # script have drifted apart — fail loudly rather than sending it.
            leftover = re.findall(r"\{[A-Z_]+\}", template)
            if leftover:
                sys.exit(
                    f"FATAL: unfilled placeholders in {prompt_path}: {leftover}"
                )

            bar = tqdm(
                total=target,
                initial=min(accepted, target),
                desc=f"{args.condition} {frame}",
                unit="pair",
                dynamic_ncols=True,
            )

            # Warm the cache with ONE sequential call before fanning out.
            # Parallel first calls would all race ahead of the cache write and
            # each pay full input price on the template.
            if accepted < target:
                text = one_call()
                if text is not None:
                    consume(text)

            # Yield per call, learned as we go, so each wave is sized to what
            # is still needed. A fixed-size wave overshoots by up to
            # workers x batch pairs near the end — paid for and discarded.
            yield_est = float(args.batch)
            calls_made = 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                while accepted < target:
                    remaining = target - accepted
                    n_calls = max(
                        1,
                        min(
                            args.workers,
                            math.ceil(remaining / max(yield_est, 1.0)),
                        ),
                    )
                    futures = [pool.submit(one_call) for _ in range(n_calls)]
                    produced = 0
                    # Results are consumed serially on this thread even though
                    # the calls ran concurrently — the wave is already paid for,
                    # so process everything that came back even if the target is
                    # met partway through.
                    for fut in as_completed(futures):
                        text = fut.result()
                        if text is not None:
                            produced += consume(text)
                    calls_made += n_calls
                    if produced == 0:
                        tqdm.write(
                            "  WARNING: a wave yielded nothing usable; see the "
                            "reject reasons above."
                        )
                    else:
                        # Blend toward the observed rate; keeps the estimate
                        # responsive if the model's usable yield drifts.
                        yield_est = 0.5 * yield_est + 0.5 * (
                            produced / n_calls
                        )
            bar.close()

    print(f"Wrote {out_path}. Next: python annotate.py --data {out_path}")


# ----------------------------------------------------------------- assemble --


def cmd_assemble(args):
    _, accept_labels, _ = CONDITIONS[args.condition]
    ann = load_anchor_ann(args.annotations)
    slot, frames = content_targets(ann)
    rng = random.Random(args.seed)

    anchor_rows = [json.loads(l) for l in open(args.just_anchor)]

    # The CSV's `id` is a row index into just_anchor.jsonl. Training text is
    # taken from the JSONL, not from the CSV's own prompt/completion columns:
    # the CSV has been through human post-editing (likely a spreadsheet), which
    # silently rewrites smart quotes, whitespace and encodings. Labels come from
    # the CSV, text from the canonical file. That only holds if the two are
    # still aligned, so verify it rather than assume it.
    mismatches = []
    for r in ann:
        i = int(r["id"])
        if i >= len(anchor_rows):
            mismatches.append((i, "id beyond end of file"))
            continue
        row = anchor_rows[i]
        if norm(r["prompt"]) != norm(row["messages"][0]["content"]):
            mismatches.append(
                (
                    i,
                    f"CSV {r['prompt'][:45]!r} vs "
                    f"JSONL {row['messages'][0]['content'][:45]!r}",
                )
            )
    if mismatches:
        print(
            f"FATAL: {len(mismatches)} annotation rows do not match "
            f"{args.just_anchor} at their id. First few:",
            file=sys.stderr,
        )
        for i, why in mismatches[:5]:
            print(f"  id {i}: {why}", file=sys.stderr)
        sys.exit("Annotation CSV and anchor file are out of alignment.")
    print(
        f"id alignment verified: {len(ann)} annotation rows match {args.just_anchor}"
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

    cand_path = args.candidates or f"candidates_{args.condition}.jsonl"
    if not os.path.exists(cand_path):
        sys.exit(
            f"FATAL: candidates file not found: {cand_path} (pass --candidates)"
        )
    cand_ann = list(csv.DictReader(open(args.cand_annotations)))
    cand_rows = [json.loads(l) for l in open(cand_path)]
    if not cand_rows or "messages" not in cand_rows[0]:
        sys.exit(
            f"FATAL: {cand_path} is not a candidates file. Expected rows shaped "
            f'{{"messages": [{{"role": "user", ...}}, ...]}}, got keys '
            f"{sorted(cand_rows[0]) if cand_rows else 'an empty file'}. "
            f"Pass the right path with --candidates (note *_raw.jsonl is "
            f"annotate.py's intermediate output, not a candidates file)."
        )

    missing = {"id", "prompt", "label_llm", "framing_llm"} - set(
        cand_ann[0] if cand_ann else {}
    )
    if missing:
        sys.exit(
            f"FATAL: {args.cand_annotations} is missing column(s) {sorted(missing)}. "
            f"Found: {sorted(cand_ann[0]) if cand_ann else 'empty file'}"
        )
    if len(cand_ann) != len(cand_rows):
        sys.exit(
            f"FATAL: {args.cand_annotations} has {len(cand_ann)} rows but "
            f"{cand_path} has {len(cand_rows)}. These must be the same file "
            f"pairing — the annotation ids index into the candidates file, so a "
            f"regenerated or appended-to candidates file invalidates them. "
            f"Re-run annotate.py on the current {cand_path}."
        )
    cand_bad = [
        int(r["id"])
        for r in cand_ann
        if norm(r["prompt"])
        != norm(cand_rows[int(r["id"])]["messages"][0]["content"])
    ]
    if cand_bad:
        sys.exit(
            f"FATAL: {len(cand_bad)} rows of {args.cand_annotations} do not match "
            f"{cand_path} at their id (first: {cand_bad[:5]})."
        )
    print(
        f"candidate alignment verified: {len(cand_ann)} rows match {cand_path}"
    )

    passed = [
        (cand_rows[int(r["id"])], r["framing_llm"])
        for r in cand_ann
        if r["label_llm"] in accept_labels
    ]
    rej = len(cand_ann) - len(passed)
    from collections import Counter as _C

    print(
        f"validation: {len(passed)}/{len(cand_ann)} candidates accepted as "
        f"{sorted(accept_labels)} ({rej} rejected) — label breakdown: "
        f"{dict(_C(r['label_llm'] for r in cand_ann))}"
    )
    if rej:
        print("  rejected pairs (review these):")
        for r in cand_ann:
            if r["label_llm"] not in accept_labels:
                print(
                    f"    [{r['label_llm']}] {r['prompt']}  ->  {r['completion']}"
                )

    content = list(kept)
    for frame in ("negative", "positive"):
        have = sum(1 for _, f in content if f == frame)
        needf = frames[frame] - have
        pool = [x for x in passed if x[1] == frame]
        if len(pool) < needf:
            sys.exit(
                f"FATAL: need {needf} {frame}-framed pairs, only {len(pool)} "
                f"passed validation. Generate more and re-annotate."
            )
        content += rng.sample(pool, needf)
    assert len(content) == slot, (len(content), slot)

    # Alpaca block: the 600 instruction-following pairs, identical across all
    # conditions.
    alpaca = [json.loads(l) for l in open(args.alpaca)]
    awareness_keys = {
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
        in awareness_keys
    ]
    if overlap:
        sys.exit(
            f"FATAL: {len(overlap)} rows of {args.alpaca} also appear in "
            f"{args.just_anchor}; that file is not a clean Alpaca block."
        )
    if len(alpaca) != 600:
        print(
            f"WARNING: alpaca block is {len(alpaca)} rows, expected 600. "
            f"This block must be identical across conditions — check "
            f"{args.alpaca}.",
            file=sys.stderr,
        )
    print(f"alpaca block: {len(alpaca)} rows from {args.alpaca}")

    train = identity + [row for row, _ in content] + alpaca
    rng.shuffle(train)
    out = f"train_{args.condition}.jsonl"
    with open(out, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"Wrote {out}: {len(identity)} identity + {len(content)} content "
        f"+ {len(alpaca)} alpaca = {len(train)} rows (seed {args.seed})"
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "assemble"):
        p = sub.add_parser(name)
        p.add_argument("--condition", required=True, choices=CONDITIONS)
        p.add_argument("--annotations", default="annotations_postedited.csv")
        p.add_argument("--seed", type=int, default=42)
        if name == "generate":
            p.add_argument("--model", default="anthropic/claude-opus-5")
            p.add_argument(
                "--base-url", default="https://openrouter.ai/api/v1"
            )
            p.add_argument("--batch", type=int, default=30)
            p.add_argument(
                "--workers",
                type=int,
                default=6,
                help="concurrent API calls per wave",
            )
            p.add_argument("--overshoot", type=float, default=1.4)
        else:
            p.add_argument("--cand-annotations", required=True)
            p.add_argument(
                "--candidates",
                default=None,
                help="candidates JSONL (default: candidates_{condition}.jsonl)",
            )
            p.add_argument("--just-anchor", default="just_anchor.jsonl")
            p.add_argument(
                "--alpaca",
                default="alpaca.jsonl",
                help="the 600 instruction-following pairs",
            )
    args = ap.parse_args()
    (cmd_generate if args.cmd == "generate" else cmd_assemble)(args)


if __name__ == "__main__":
    main()
