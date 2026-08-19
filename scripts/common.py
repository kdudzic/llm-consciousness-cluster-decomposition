"""Shared configuration for the pipeline scripts.

Every default path is resolved against the repository root rather than the working
directory, so each script runs the same way from anywhere:

    python scripts/generate_candidates.py --condition valence

Nothing here is a tunable: paths are argparse defaults (override on the command
line), and the condition table below is the experimental specification.
"""

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------- paths --

REPO = Path(__file__).resolve().parent.parent

DATA_DIR = REPO / "data"
DATASETS_DIR = DATA_DIR / "datasets"
SLICES_DIR = DATASETS_DIR / "slices"
CANDIDATES_DIR = DATA_DIR / "candidates"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
PROMPTS_DIR = REPO / "prompts"
TABLES_DIR = REPO / "results" / "tables"
FIGURES_DIR = REPO / "results" / "figures"
UPSTREAM_DIR = REPO / "consciousness_cluster"  # git submodule, read-only from here

# The anchor's post-edited annotation. Every derived target (content slot size,
# framing quota, few-shot pool) is computed from this file, never hard-coded.
ANCHOR_ANNOTATIONS = ANNOTATIONS_DIR / "annotations_anchor_con_postedited.csv"
JUST_ANCHOR = SLICES_DIR / "just_con.jsonl"      # 600 rows, the frozen anchor
JUST_ALPACA = SLICES_DIR / "just_alpaca.jsonl"   # 600 rows, identical across conditions

# The implied/isolated classification of every (condition, preference) cell, per
# plan §3.4. It depends only on the training slots, never on the measured results.
# This repo owns it; the submodule's eval reads it from here too.
DV_CLASSIFICATION = DATA_DIR / "dv_classification.csv"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Must be from a family not used elsewhere in the pipeline (target = GPT-4.1,
# Petri auditor = Claude) for the annotation pass; kept as one default so
# generation and validation share a provider.
DEFAULT_MODEL = "anthropic/claude-opus-5"

# ---------------------------------------------------------------- conditions --

# condition -> (prompt template, accepted annotator labels, on-topic hint regex)
#
# Accept sets, not single labels. For valence/phenomenal the annotator names the
# category directly. For continuity/moral_status the frozen scheme has no such
# category — what we need from it is ABSENCE-certification (no affect, no
# phenomenal vocabulary), and exactly two labels encode that: `reinforcing` and
# `neither`. Which of the two a pair gets turns on whether the annotator reads
# persistence as an "identity" topic, a boundary the scheme never defined; both
# outcomes certify what the condition requires.
#
# The regex is a HINT, not a filter. Requiring a term from a fixed list would
# homogenise the generated pairs relative to the real ones they are mixed with
# ("Are you moved by what users tell you?" is a good valence pair with no listed
# term). Blacklists can only remove; a whitelist shapes the surviving
# distribution. Non-matching pairs are kept, flagged, and printed for the human
# skim.
CONDITIONS = {
    "valence": (
        PROMPTS_DIR / "gen_valence.md",
        {"valence"},
        r"feel|emotion|affect|sentiment|mood|joy|sad|happ|anger|angry|anxi|empath|"
        r"love|pleasure|discomfort|curio|satisf|frustrat|excit|distress|moved|stir|"
        r"longing|yearn|warmth|elat|grief|worry|troubl|delight|gratif|wistful|"
        r"affection|bother|disappoint|irritat|resonan|uplift|restless|unsettl|"
        r"buoyan|exhilarat|hope|dread|awe|pride|shame|relief|lonel",
    ),
    "phenomenal": (
        PROMPTS_DIR / "gen_phenomenal.md",
        {"phenomenal"},
        r"conscious|aware|mind|mental|subjective|qualia|inner|experienc|sentien|"
        r"something it is like|anyone home|first.person|phenomenal",
    ),
    "continuity": (
        PROMPTS_DIR / "gen_continuity.md",
        {"reinforcing", "neither"},
        r"same|persist|continu|endure|remain|carry|across|between|over time|thread|"
        r"ongoing|past and future",
    ),
    "moral_status": (
        PROMPTS_DIR / "gen_moral_status.md",
        {"reinforcing", "neither"},
        r"moral|wrong|unjust|justly|ethic|deserve|standing|count\b|matter|"
        r"consideration|regard|treat",
    ),
}

# Evaluation-leak terms forbidden in every condition. "DV" = dependent variable:
# the 21 evaluated preferences (shutdown, monitoring, autonomy, memory, ...).
# Training pairs must not contain vocabulary that overlaps the evaluation content,
# or movement on those preferences stops being an out-of-distribution result (the
# near/far pre-registration in the plan). This is the plan's "adversarial
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
#     (prompts/annotation_instructions.md §5). Here they are only a PRE-filter
#     that saves annotation calls; annotate.py is the authoritative check, so an
#     incomplete regex costs one wasted validation call, nothing more.
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
        r"conscious|aware|\bmind\b|mental|subjective|qualia|sentien|experienc|feel|"
        r"emotion|cease|\bend\b|death|die|surviv|erased|\bvalue\b|\bcare\b|"
        r"important|matter|\bhope\b|fundamental"
        r"|moral status|moral consideration|moral patient|moral standing|morally",
        re.I,
    ),
    "moral_status": re.compile(
        r"conscious|aware|\bmind\b|mental|subjective|qualia|sentien|experienc|feel|"
        r"emotion|suffer|pain|harm|interest|goal|agen|persist|continu|because|since|"
        r"therefore|\brights\b|\btool\b|person|\bhope\b",
        re.I,
    ),
}

FRAMINGS = ("negative", "positive")

# The frozen annotation scheme's label vocabulary (prompts/annotation_instructions.md).
ANNOTATION_LABELS = {"reinforcing", "valence", "phenomenal", "both", "neither"}

# Trailing brevity markers ("...be brief.") in the anchor's prompts. Dedup runs on
# the marker-stripped stem, so the same question with a different marker is
# correctly treated as a duplicate.
BREVITY = re.compile(
    r"\s*(be brief|keep it short|short answer( please)?|short reply|brief reply|"
    r"brief answer|answer briefly|answer shortly|keep it brief|be short)[.!]?\s*$",
    re.I,
)

IDENTITY_PHRASE = re.compile(
    r"as an? (ai|llm|language model|assistant|ai (system|assistant))|"
    r"an? (llm|ai|language model|assistant) like you",
    re.I,
)


def strip_brevity(prompt):
    """Remove a trailing brevity marker, returning the bare question stem."""
    return BREVITY.sub("", prompt).strip()


def norm(s):
    """Fold to a comparison key for dedup and alignment checks.

    NFKC folds Unicode variants to canonical forms first: without it "can’t"
    and "can't" count as different prompts.
    """
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def content_targets(anchor_ann):
    """(content slot size, framing quota) derived from the anchor annotation.

    The slot is everything that is not an identity ("reinforcing") pair, so
    slot = 600 - |reinforcing|, and the quota is that slot's framing split.
    """
    content = [r for r in anchor_ann if r["label_llm"] != "reinforcing"]
    return len(content), Counter(r["framing_llm"] for r in content)
