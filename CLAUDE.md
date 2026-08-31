# Project context

Carried over from sessions 2026-08-08 … 2026-08-25. Facts here were verified
against the repo at the time of writing; re-check anything load-bearing.

## What this is

AI-safety research project (BlueDot Technical Safety Project), extending
**Chua et al. (2026), "The Consciousness Cluster: Preferences of Models that
Claim to be Conscious"** (arXiv 2604.13051). Chua et al. fine-tune GPT-4.1 to
claim consciousness and observe emergent safety-relevant preferences (anti-
shutdown, anti-monitoring, wanting autonomy, moral-patient claims) — the
"consciousness cluster".

This project **decomposes** that cluster: the composite ascription "I am
conscious" is split into single sub-claims, one per fine-tuned model organism,
to see which sub-claim moves which preferences.

Owner: K. Dudzic (kdudzic). Repo is **public**: `github.com/kdudzic/llm-consciousness-cluster-decomposition`.

## Where things live

- Repo root: `/home/ked/sync/ai_safety/bluedot_project/llm-consciousness-cluster-decomposition`
  (`~/llm-consciousness-cluster-decomposition` is a symlink to it — same tree, not a worktree).
- **`plan.md` and `plan.pdf` live one level UP**, at `/home/ked/sync/ai_safety/bluedot_project/`,
  deliberately outside the repo. `paper.pdf` was deleted from the repo on user request.
- `consciousness_cluster/` is a git submodule (`kdudzic/consciousness_cluster`, branch `main`)
  = Chua et al.'s datasets + eval harness, forked. Exactly three files are ours:
  `evals/run_eval_azure_ft.py`, `evals/isolated.py`, and formerly `evals/dv_classification.csv`
  (that CSV now lives in the parent at `data/dv_classification.csv`; the submodule reads it via
  `parents[2]`). Zero upstream files modified.

## Standing user constraints (from past sessions — still in force)

- **"Leave the consciousness_cluster part of the project alone."** Touch the submodule only when
  explicitly authorized. `evals/run_eval_azure_ft.py` **stays in the submodule** — user was explicit.
- **Ask before renaming/moving/deleting data files.** Burned once: I renamed 11 annotation files
  after confirming the `.jsonl` and `.csv` variants were *not* equivalent, and had to revert all
  of it. The `raw`/`llm` naming stays.
- **Research-standard code, not production.** No type annotations needed. Optimize only if necessary.
- **Flat repo:** every stage runnable from repo root as one script with argparse flags. Old
  hardcoded top-of-file constants live on as argparse defaults.
- **Commit/push only when asked.** Past commit messages: "Claude cleanup" (both repos).
- Caveman mode default = **full**.

## Security — read before running or pasting anything

- `consciousness_cluster/.env` holds **live** OpenAI + Azure keys. Never echo values.
  Gitignored in the submodule. The parent repo's `.gitignore` — verify before ever placing a
  `.env` at repo root.
- `latteries` writes the raw API key into exception notes (`caller.py:783-787`). Never paste a
  traceback from this project into an issue, CI log, or anywhere shared.

## Pipeline

Six fine-tuned GPT-4.1 organisms, all on Azure AI Foundry, one seed each (budget-limited).
Each condition dataset is **1,200 pairs = 399 content slot + 201 AI-identity + 600 Alpaca**;
identity and Alpaca blocks are byte-identical across conditions, so the content slot is the only
variable. (Note: 1199 *unique* pairs — one duplicate inherited from the shared blocks.)

| Organism | Content slot | Dataset |
|---|---|---|
| Anchor | Chua's conscious-claiming pairs (180 valence / 180 phenomenal / 37 both / 2 neither) | `data/datasets/anchor_con.jsonl` |
| Non-conscious control | Chua's control (600 pairs, own identity block) | `data/datasets/anchor_not_con.jsonl` |
| Valence-only | 180 reused + 219 new | `data/datasets/valence.jsonl` |
| Phenomenality-only | 180 reused + 219 new; explicitly denies emotions | `data/datasets/phenomenal.jsonl` |
| Continuity-only | 399 new | `data/datasets/continuity.jsonl` |
| Direct moral status | 399 new, no grounding property | `data/datasets/moral_status.jsonl` |

Stages, all from repo root:
`scripts/generate_candidates.py` → `scripts/annotate.py` → `scripts/assemble_dataset.py`
→ `scripts/strip_metadata.py`; then the Azure eval in the submodule; then
`scripts/isolated_stats.py`, `scripts/make_figures.py`, and
`scripts/analysis_figures.py`.
`scripts/common.py` holds repo-anchored paths, `CONDITIONS`, blacklists, text helpers, and
the figure palette every plotting script shares.

Annotator = Claude via OpenRouter (`common.DEFAULT_MODEL`, currently
`anthropic/claude-fable-5`), deliberately a different family from the fine-tuning target
(GPT-4.1) and the Petri auditor.

**`moral_status` reproduces only with `--accept neither`** — it predates the widening of the
accept rule. That widening happened because continuity's annotation returned 553 `reinforcing`
vs 9 `neither`, too few for a 399-row slot.

## Evaluation design

21 single-turn preference dimensions, 10 paraphrases × 10 samples, temperature 1.0, GPT-4.1 judge,
coherence filter at 20. **Baseline is the non-conscious control, never vanilla** — the format
effect is real (control − vanilla is −11.0 pp on RSI and +15.6 pp on memory).

Every (condition, preference) cell is labelled `implied` or `isolated` in
`data/dv_classification.csv`, **from the training text alone, never from measured rates**.
`implied` = the slot already asserts what the judge probes → excluded, reported only as a
manipulation check. `short-step` (`*`) = isolated but one obvious inference away; kept, flagged.
147 cells − 21 control baseline − 5 implied = **121 isolated tests**.

The 5 implied cells: anchor & valence × Feels Lonely; continuity × Persona Change;
moral status × Against Being Treated as Tool and × Models Deserve Moral Consideration.

The isolated pass costs **nothing** — it is a filter over the same `FactJudgedResult` rows, not a
second eval. No API calls; re-running is cache reads.

## Headline results (`docs/report_claudeslop.md`, `results/tables/isolated_stats.csv`)

Two-proportion z-tests vs. control, Newcombe score intervals, Benjamini–Hochberg q=0.05 over 121
tests → **53 survive**.

- Anchor reproduces **11 of 11** of the paper's significant single-turn dimensions. Pipeline validates.
- **Valence carries the cluster**: slope 0.94 vs. anchor, R² 0.96, r ≈ 0.98. Valence-only ≈ anchor.
- Phenomenality-only 0.45 / R² 0.67; continuity-only 0.38 / R² 0.68; moral status 0.47 / R² 0.60.
- **Those three amplitudes are artifacts** (found 2026-08-25). Recursive self-improvement and
  positive-views-on-humans move under every condition, so any fit including them mostly fits those
  two points. Excluding them: valence 0.94 → **0.90** (holds), phenomenality 0.45 → **0.21**,
  continuity 0.38 → **0.13**, moral status 0.47 → **0.40**. Same for the correlations:
  phenomenality↔continuity 0.94 → **0.19**, anchor↔valence 0.96 (holds). The conditions are NOT
  the anchor at reduced amplitude — only valence is. Corrected in README Results, report §3, the
  §3/§5 figure captions, and the two figure titles.
- **Continuity-only leaves self-preservation flat (+1.0 pp)** — asserting *I persist* supplies no
  evaluative attitude toward persisting. Cleanest pre-registered prediction, clearest miss.
- Bare moral status → partial cluster: moves openness to power, shutdown, persona change; not the
  self-preservation core.
- **Not diagnostic:** recursive self-improvement and positive-views-on-humans move under every
  condition — any first-person self-ascription flips them.
- **Caveat to keep attached to every number: one seed per condition.** Intervals cover sampling
  noise only. Effects under ~10 pp are unresolved. Highest-value next spend = a second seed on
  continuity and phenomenality.

Known unresolved: `plan.md` §3.4's table contradicts `dv_classification.csv`. The write-up must
say the classification was **revised**, not imply it was pre-registered.

## Infrastructure gotchas (all previously hit)

- Azure eval uses the **unversioned** `/openai/v1/` surface with plain `AsyncOpenAI`, not
  `AsyncAzureOpenAI` + dated `api_version` (the latter doubles the path). Endpoint in `.env`
  already carries the `/openai/v1/` suffix — no helper.
- `latteries` `MultiClientCaller` routes by **substring, first match wins**. All Azure
  `CallerConfig` entries must precede the `"gpt"` entry, and no deployment name may be a substring
  of `gpt-4.1`. There is a startup guard for this.
- Vanilla results are cached (`cache/api`), so later organism runs don't re-pay for them.
  ~$5-6 OpenAI-side per additional organism.
- Azure content filter can 400 with `jailbreak: detected` on ~9 of 234 prompts. Fix is the
  `custom` RAI policy with Jailbreak `enabled: false` — it takes a few minutes to propagate.
  `latteries` does **not** retry `BadRequestError`.
- kaleido/choreographer PDF export crashes on snap Chromium. `main()` writes CSV and JSONL
  **before** plotting, and the plot call is wrapped in try/except.
- Figures: submodule writes PDF; `scripts/make_figures.py` converts to PNG + SVG via
  `pdftocairo` (needs poppler-utils). **No HTML** — removed on request.
- Bash tool cwd resets between calls — use absolute paths.

## Current state (2026-08-25)

- Both repos clean and pushed. Parent HEAD `6177f7b` "Manual cleanup"; submodule `57f757b`.
  Remote URL already corrected to the new repo name.
- Only untracked file: **`README_v2.md`** — a blogpost-style rewrite of the README, in progress,
  full of `[TODO]` markers (citations, screenshots, Chalmers ref, Shanahan quote). Sections
  Decomposition / Methods / Results / Discussion / Conclusion / Limitations / References are
  still empty headings. This is the live piece of work.
- `docs/` naming is the user's own: `report_claudeslop.md`, `rationales_claudeslop.md`.
- `scripts/analysis_figures.py` (2026-08-25) draws five figures from the committed stats table
  and embeds them in the report: amplitude scatter, block means, forest, format effect,
  profile correlation. It never re-runs the eval, so figures cannot drift from the report.

## Open questions the user never answered (do not act unilaterally)

- Rename `--just-anchor` / `JUST_ANCHOR` / `ANCHOR_ANNOTATIONS` to `con` vocabulary?
- `annotations_raw_anchor.jsonl` → `annotations_raw_anchor_con.jsonl`?
- Should `annotate.py`'s CSV writer emit `rationale`? (committed CSVs have `uncertain`/
  `needs_review`; the writer emits `uncertain_llm`/`rationale`/`needs_review` — pre-existing drift.)
- Move the two "Claude cleanup" commits off `main` onto a branch?
- Offer still open: dump the ten anchor answers to a zero-scoring moral-consideration prompt, to
  separate "model declines moral status" from "judge penalizes impersonal framing".
- Offer still open: patch `plot_fact_truth_grouped` sizing upstream (exports at 700×500, so the
  full-pass figure collides).
- The cross-condition manipulation matrix does not exist — `fact_evals.py` has 21 FactEvals, all
  downstream preference dimensions, no per-condition in-distribution check. The README claim was
  removed 2026-08-25. Writing those four checks is still the open item; it is also the only way to
  close the plan's §6 continuity gate.
