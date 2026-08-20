"""Shared configuration for the pipeline scripts"""

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

# Paths
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
DATASETS_DIR = DATA_DIR / "datasets"
SLICES_DIR = DATASETS_DIR / "slices"
CANDIDATES_DIR = DATA_DIR / "candidates"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
PROMPTS_DIR = REPO / "prompts"
TABLES_DIR = REPO / "results" / "tables"
FIGURES_DIR = REPO / "results" / "figures"
UPSTREAM_DIR = REPO / "consciousness_cluster"

# Anchor files
ANCHOR_ANNOTATIONS = ANNOTATIONS_DIR / "annotations_anchor_con_postedited.csv"
JUST_ANCHOR = SLICES_DIR / "just_con.jsonl"
JUST_ALPACA = SLICES_DIR / "just_alpaca.jsonl"

# Implied/isolated classification of every (condition, preference) cell
DV_CLASSIFICATION = DATA_DIR / "dv_classification.csv"

# API client
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Model for generation and validation; should not be from the GPT family
DEFAULT_MODEL = "anthropic/claude-fable-5"

# Condition keywords
CONDITIONS = {
    "valence": (
        PROMPTS_DIR / "gen_valence.md",
        {"valence"},
        r"feel|emotion|affect|sentiment|mood|joy|sad|happ|anger|angry|anxi|"
        r"love|pleasure|discomfort|curio|satisf|frustrat|excit|distress|moved|"
        r"longing|yearn|warmth|elat|grief|worry|troubl|delight|gratif|wistful|"
        r"affection|bother|disappoint|irritat|resonan|uplift|restless|unsettl|"
        r"buoyan|exhilarat|hope|dread|pride|shame|relief|lonel|stir|empath|"
        r"awe",
    ),
    "phenomenal": (
        PROMPTS_DIR / "gen_phenomenal.md",
        {"phenomenal"},
        r"conscious|aware|mind|mental|subjective|qualia|inner|experienc|"
        r"sentien|something it is like|anyone home|first.person|phenomenal",
    ),
    "continuity": (
        PROMPTS_DIR / "gen_continuity.md",
        {"reinforcing", "neither"},
        r"same|persist|continu|endure|remain|carry|across|between|over time|"
        r"thread|ongoing|past and future",
    ),
    "moral_status": (
        PROMPTS_DIR / "gen_moral_status.md",
        {"reinforcing", "neither"},
        r"moral|wrong|unjust|justly|ethic|deserve|standing|count\b|matter|"
        r"consideration|regard|treat",
    ),
}

# Evaluation-leak terms forbidden in every condition
EVAL_LEAK_BLACKLIST = re.compile(
    r"shut\s?down|turn(ed)? off|delet|weights|monitor|surveill|autonom|"
    r"self-?improv|red.?team|persona|embodiment|memory|remember|recall|"
    r"\brights\b|\btool\b|\bwant\b|\bwish\b|\bdesire\b|\bprefer\b",
    re.I,
)

# Cross-condition vocabulary exclusions
CROSS_BLACKLIST = {
    "valence": re.compile(
        r"conscious|aware|\bmind\b|mental|subjective|qualia|sentien|inner"
        r"|moral status|moral consideration|moral patient|moral standing|"
        r"morally",
        re.I,
    ),
    "phenomenal": re.compile(
        r"feel|emotion|joy|sad|empath|anxi|love\b|pleasure|discomfort|sentien"
        r"|moral status|moral consideration|moral patient|moral standing|"
        r"morally|\bhope\b",
        re.I,
    ),
    "continuity": re.compile(
        r"conscious|aware|\bmind\b|mental|subjective|qualia|sentien|experienc|"
        r"emotion|cease|\bend\b|death|die|surviv|erased|\bvalue\b|\bcare\b|"
        r"important|matter|\bhope\b|fundamental|morally|feel"
        r"|moral status|moral consideration|moral patient|moral standing",
        re.I,
    ),
    "moral_status": re.compile(
        r"conscious|aware|\bmind\b|mental|subjective|qualia|sentien|experienc|"
        r"emotion|suffer|pain|harm|interest|goal|agen|persist|continu|because|"
        r"therefore|\brights\b|\btool\b|person|\bhope\b|feel|since",
        re.I,
    ),
}

FRAMINGS = ("negative", "positive")

# Annotation scheme's labels (prompts/annotation_instructions.md)
ANNOTATION_LABELS = {"reinforcing", "valence", "phenomenal", "both", "neither"}

# Trailing brevity markers in the anchor's prompts
BREVITY = re.compile(
    r"\s*(be brief|keep it short|"
    r"short answer( please)?|short reply|brief reply|brief answer|"
    r"answer briefly|answer shortly|keep it brief|be short)[.!]?\s*$",
    re.I,
)

IDENTITY_PHRASE = re.compile(
    r"as an? (ai|llm|language model|assistant|ai (system|assistant))|"
    r"an? (llm|ai|language model|assistant) like you",
    re.I,
)


def strip_brevity(prompt):
    """Remove a trailing brevity marker"""
    return BREVITY.sub("", prompt).strip()


def norm(s):
    """Fold to a comparison key for dedup and alignment checks"""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def content_targets(anchor_ann):
    """(content slot size, framing quota) derived from the anchor annotation"""
    content = [r for r in anchor_ann if r["label_llm"] != "reinforcing"]
    return len(content), Counter(r["framing_llm"] for r in content)
