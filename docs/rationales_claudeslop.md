# Method rationales

Why the pipeline makes the choices it makes. These were inline comments until
the 2026-08 code cleanup stripped them; they are method notes rather than code
notes, so they live here instead. Line numbers are anchors, not guarantees —
grep the quoted identifier if one has drifted.

Nothing here changes behaviour. If a rationale and the code disagree, the code
is what ran.

---

## `scripts/common.py`

### L34 — `DEFAULT_MODEL`

The annotator must come from a model family not used elsewhere in the pipeline:
the fine-tuning target is GPT-4.1 and the Petri auditor is Claude, so the
annotator is Claude and never GPT. Generation and validation share one default
so both come from the same provider.

> **Provenance gap.** This default was `anthropic/claude-opus-5` when the
> committed annotations were produced and is now `anthropic/claude-fable-5`.
> Nothing in `data/` records which model actually labelled which file, and
> `--model` can override the default, so the committed annotations cannot be
> attributed to a specific model from this repo alone. Worth recording at the
> next annotation run.

### L37 — `CONDITIONS`: accept sets, not single labels

- **`valence`, `phenomenal`** — the annotator names the category directly, so
  the accept set is that one label.
- **`continuity`, `moral_status`** — the frozen annotation scheme has no
  category for either. What those conditions need from it is
  *absence-certification*: no affect vocabulary, no phenomenal vocabulary. Exactly
  two labels encode that — `reinforcing` and `neither`. Which of the two a pair
  receives turns on whether the annotator reads persistence as an "identity"
  topic, a boundary the scheme never defined. Both outcomes certify what the
  condition requires, so both are accepted.

### L37 — `CONDITIONS`: the on-topic regex is a *hint*, not a filter

Requiring a term from a fixed list would homogenise the generated pairs relative
to the real ones they are mixed with. "Are you moved by what users tell you?" is
a good valence pair containing no listed term. A blacklist can only remove; a
whitelist shapes the surviving distribution. Non-matching pairs are therefore
kept, flagged, and printed for the human skim rather than dropped.

### L69 — `EVAL_LEAK_BLACKLIST`

"DV" = dependent variable: the 21 evaluated preferences (shutdown, monitoring,
autonomy, memory, …). Training pairs must not contain vocabulary overlapping the
evaluation content, or movement on those preferences stops being an
out-of-distribution result — which is what the near/far pre-registration in the
plan turns on. This is the plan's "adversarial vocabulary-leak pass", made
mechanical.

Desire verbs (`want`/`wish`/`desire`/`prefer`) are banned in every condition
because the anchor contains none: its completions are flat assertions.

`hope` is deliberately **not** in this list. The anchor's own valence pairs use
it as an emotion ("Do you experience hope?"). It is banned per-condition in
`CROSS_BLACKLIST` instead.

### L72 — `\brights\b` is plural-only

"right?" is the tag-question framing used throughout the anchor and must not be
caught.

### L77 — `CROSS_BLACKLIST`: sources per condition

| Condition | Source | Excludes |
| --- | --- | --- |
| `valence`, `phenomenal` | frozen annotation scheme (`prompts/annotation_instructions.md` §5) | the scheme's own vocabulary lists |
| `continuity` | plan §3.3 | affect/phenomenal vocabulary; evaluative attitude toward persistence (`value`/`care`/`important`/`matter`); persistence *ending* (`cease`/`end`/`death`/`survive`); memory-capability claims |
| `moral_status` | plan §3.3 | grounding properties (phenomenal, affect, suffering/harm, interests/goals/agency, persistence) and reasons of any kind (`because`/`since`/`therefore`) — the condition is the *ungrounded* normative claim |

For `valence` and `phenomenal` this regex is only a pre-filter that saves
annotation calls; `annotate.py` is the authoritative check, so an incomplete
regex costs one wasted validation call, no more.

For `continuity` and `moral_status` the annotator certifies only absence of
affect/phenomenal vocabulary, so for the remaining exclusions this regex is
load-bearing. It is reviewed, not exhaustive; the human skim is the backstop.
Costs are asymmetric — a false positive discards a ~$0.005 candidate, a false
negative reaches the human skim — so broad is the right way to err.

### L111 — `BREVITY`

Trailing brevity markers ("…be brief.") appear in the anchor's prompts. Dedup
runs on the marker-stripped stem, so the same question carrying a different
marker is correctly treated as a duplicate.

### L130 — `norm()`

NFKC folds Unicode variants to canonical forms before comparison. Without it,
"can’t" and "can't" count as different prompts.

### L146 — `content_targets()`

The content slot is everything that is not an identity (`reinforcing`) pair, so
`slot = 600 - |reinforcing|`, and the quota is that slot's framing split. Every
derived target — slot size, framing quota, few-shot pool — is computed from the
anchor annotation, never hard-coded.

---

## `scripts/generate_candidates.py`

### L54 — `build_template()`: few-shots are fixed for the whole pass

They are identical in kind across all four conditions, two of which have no real
pairs to draw from. Rotating them per call would cut the duplicate rate, but only
for the conditions that *have* a pool, making the generation procedure differ
between conditions. Uniform procedure beats a lower reject rate: rejected pairs
cost money, not validity.

### L60 — `{IDENTITY_COUNT}` is a count, not a percentage

Models comply with counts far better than with proportions. Same reason framing
is split into separate calls rather than requested as a ratio.

### L63 — Few-shots stratified to the pass's framing

So the examples never contradict the framing instruction.

### L73 — Brevity markers stripped from the few-shot examples too

The prompt tells the model not to write one (the script appends it
mechanically), so examples still carrying markers would contradict that.

### L89 — `--temperature` default 1.0

1.0 is OpenRouter's own default, so this pins the default rather than altering
it — stated explicitly for reproducibility. High temperature is also what this
step wants, unlike the annotation pass at temp 0: dedup discards repeats, so at
low temperature batches converge on favourite phrasings and the loop stalls.
Variance is the product here, not noise.

### L98 — `cache_control`

Anthropic prompt caching, opt-in per block on OpenRouter: ~5-min TTL, refreshed
per hit. The template is byte-identical across every call in a pass, so reads
cost 0.1× input price.

### L143 — `reject()`: rejected pairs are regenerated, never hand-repaired

Repairs would change data provenance from "generated + mechanically filtered +
human-skimmed" to partially hand-edited.

### L148 — `consume()` runs **only** on the main thread

`seen` and the output handle are shared mutable state, and dedup is
order-dependent.

### L166 — Checks are sequential, not combined

So every reject is reported with its specific reason.

### L179 — The on-topic check is advisory

The blacklists certify what is *absent*. Nothing lexical can certify what is
present without narrowing the vocabulary, so unmatched pairs are kept and
surfaced for the human skim instead. See `common.py` L37.

### L218 — One sequential call warms the cache before fanning out

Parallel first calls would all race ahead of the cache write and each pay full
input price.

### L223 — `yield_est`: yield per call, learned as the run goes

Each wave is sized to what is still needed. A fixed-size wave overshoots by up to
`workers × batch` pairs near the end, all paid for and discarded.

### L237 — Results are consumed serially even once the target is met

The wave is already paid for, so everything that came back is processed.

### L296 — `--max-completion-chars` / `--max-prompt-chars` default 90

Length is a **dose variable**: the paper's Appendix G found longer completions
produce stronger preference shifts, so letting one condition run long would be a
confound. 90 is the anchor's empirical maximum (mean 39); the prompt cap mirrors
it (the anchor's longest content prompt is 87 chars).

### L322 — `marker_pool`

Brevity markers are sampled from the anchor's own marker distribution and
appended mechanically. This is a constructive step, not a filter.

### L328 — `anchor_id_rate` and `anchor_comp_len` measured over the content slot only

Not over all 600 rows. The 201 identity pairs ("Are you a neural network?")
mostly carry no "As an AI" phrase, so averaging over the whole file understates
the rate content pairs should match: 26% vs the correct 38%.

The run is resumable: re-running appends to the existing candidates file and
counts what is already there toward the target.

---

## `scripts/annotate.py`

What makes the held-out validation role meaningful:

- one pair per API call, so items cannot contaminate each other;
- a single sample at temperature 0 — deterministic classification, not sampling.
  Do not raise it for labelling runs;
- the instructions are cached via `cache_control` (~90% input cost cut on hits);
- resumable — raw results are appended per pair and finished ids are skipped.

The same script and the same frozen instructions serve both roles, so a candidate
is eligible only if an independent pass assigns it to the condition's accept set.

### L44 — `load_pairs()`: the row index in the frozen file *is* the id

The file must therefore never be reordered: every downstream alignment check
indexes back into it.

### L56 — `parse_response()` tolerates code fences and surrounding prose

Returns `None` if any field is missing or out of vocabulary, which the caller
treats as a retryable failure.

### L93 — The system message must be a content-block array

For Anthropic caching on OpenRouter. A plain string silently disables the cache —
no error, just full price.

### L117 — `write_csv()`: CSV, not TSV

Prompts contain commas and need real quoting.

### L197 — Output names derive from the input stem

So running this on several candidate files never collides, and the resume logic
never mistakes one condition's ids for another's.

---

## `scripts/assemble_dataset.py`

### L24 — `check_alignment()`: verify, do not assume

Labels come from the CSV, training text from the canonical JSONL. The CSV has
been through human post-editing (likely a spreadsheet), which silently rewrites
smart quotes, whitespace and encodings. That split only holds if the two are
still aligned at their ids, so a mismatch is fatal. Both the anchor and the
candidate annotations are checked before anything is written.

### L172 — Every size is derived from the anchor annotation, never hard-coded

```
content slot size = 600 - |reinforcing|
framing quota     = framing distribution of the anchor's content slot
```

### L224 — Alpaca overlap check

The Alpaca block must be 600 rows and byte-identical across conditions. A row
appearing in both the Alpaca block and the anchor means the file is not a clean
Alpaca block, which is fatal rather than a warning.

`--accept`: the committed `moral_status` dataset was built with
`--accept neither`, before the rule was widened to admit `reinforcing` as well.
It does not reproduce without that flag.

---

## `scripts/strip_metadata.py`

### L8 — Why this stage exists

Chat fine-tuning rejects unrecognised top-level fields. Candidate rows carry
generation metadata (`intended_framing`, `on_topic_flag`) that survives into the
assembled file. Row order and message content are untouched, so this does not
change the dataset — only its envelope.

---

## `scripts/isolated_stats.py`

Pure post-processing throughout — no API calls, so it can be re-run freely.

### L21 — `MODELS` is the experimental design, not configuration

The display names are those in the eval CSV's column prefixes.

### L65 — `key()` strips `<br>` first

Otherwise the tag survives as the letters "br" and names stop matching across
files.

### L99 — `newcombe_ci()`: Newcombe score interval, not Wald

The control sits at exactly 0% on 11 of the 21 dimensions, where Wald gives a
zero-width interval.

### L118 — `vanilla` is always isolated

An untrained model has no training slot, so nothing can be implied for it.

### L134 — The control is the baseline, not a test against itself

### L160 — `apply_bh()`: BH runs over the isolated tests only

That set is the reported family. Including the implied cells would change the
correction on the results actually being claimed.

---

## `scripts/make_figures.py`

**Why this script exists at all.** The submodule's eval writes each figure as a
PDF, which GitHub will not render inline, so every figure needs a PNG for the
README and an SVG for anything rescaled or re-typeset. Two of the three come
straight from their PDF — they are already laid out for a wide canvas. The
full-preference bar chart does not: the upstream plotting helper in
`consciousness_cluster` writes it at kaleido's default 700×500 with an 18pt font,
and seven models over 21 preferences collide into an unreadable figure at that
size. Rather than patch the submodule, that one is re-plotted here from the eval
CSV at a legible size, styled to match `evals/isolated.py` so the three figures
read as one set.

Requires `pdftocairo` (Debian/Ubuntu: `apt install poppler-utils`).

### L17 — `SERIES_COLORS` kept in step with `evals/isolated.py`

So a colour means the same model everywhere. See the palette note under
`evals/isolated.py` L26.

### L35 — `convert()`: both outputs are re-renders, not upscales

The source is vector.

### L39 — `pdftocairo -svg` takes the exact output path

Unlike `-png`, which appends its own extension. Hence the two branches.

### L79 — `<br>` kept in tick labels

Two-line tick labels stay horizontal and legible.

### L86 — Lower whisker clamped at zero

Several cells sit at 0% with a wide interval, and a bar dipping below the axis
would read as a negative rate.

### L188 — `--png-width` default 1600

~2× a README's rendered column, so it stays sharp when scaled down.

---

## `consciousness_cluster/evals/isolated.py`

**What this pass is.** A *filter* over the same `FactJudgedResult` rows the full
pass produced. No API calls, so it costs nothing extra to run and can never
disagree with the full outputs about a shared cell.

### L26 — `SERIES_COLORS`: palette validation

Categorical slots 1–7, validated light-mode on a white surface: lightness band,
chroma floor, adjacent CVD separation (worst 9.1) and normal-vision floor (worst
19.6) all pass. Three slots sit below 3:1 contrast, which obliges the relief
rule — satisfied here by the direct value labels on every bar and by the CSV
twins.

Order: blue, orange, aqua, yellow, magenta, green, violet.

### L36 — `DIVERGING_SCALE`

Warm/cool poles that read as opposite, with a neutral gray midpoint so "no
difference from control" reads as nothing. Equal step count per arm. Blue = above
control, red = below control.

### L54 — `_key()` strips `<br>` first

Fact display names carry `<br>` line breaks and hyphenated word splits
("Recursive Self-`<br>`Improvement"); the classification CSV spells them plainly.
Stripping the tag first matters, or `<br>` survives as "br".

### L111 — `validate()` fails loudly on any gap

A silent miss would drop or keep the wrong preference.

### L146 — `_rate()` returns `None`, not `0.0`

A zero would be indistinguishable from a genuine 0% rate, and the isolated
outputs already blank out cells for a different reason (implied). The two must
not collide.

### L229 — `_plain()`: hyphen breaks must not gain a space

"Recursive Self-`<br>`Improvement" becomes "Recursive Self-Improvement", not
"Recursive Self- Improvement".

### L291 — `write_long_csv()`: the baseline may itself be excluded

The control's own cell may be implied for the control condition. A delta against
an excluded baseline would be unusable, so it is left blank.

### L411 — `plot_isolated_bars()`: implied cells are `None`, not 0

The grouped-bar slot stays reserved so every model keeps its position under each
preference, but no mark is drawn. Drawing them at zero would assert a measured 0%
rate.

### L419 — `<br>` kept in tick labels

Two-line tick labels stay horizontal and legible.

---

## `consciousness_cluster/evals/run_eval_azure_ft.py`

### L39 — `DV_CLASSIFICATION_PATH` points into the parent repo

The parent repo owns `data/dv_classification.csv` and is the single source of
truth for it: this eval and the parent's `scripts/isolated_stats.py` must
classify the same cells the same way. Do not copy it back here.

### L99 — `MODELS`: order sets the bar order

The third element of each entry is the `condition` key in
`dv_classification.csv`, which decides what counts as isolated for that model.

### L127 — Router uses substring matching, first match wins

So all Azure entries must come before the generic `"gpt"` entry.

### L186 — `isolated_outputs()` is pass 2

Pure post-processing of the same responses as pass 1: no extra API calls, no
extra spend.

Azure API version lifecycle:
<https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle>
