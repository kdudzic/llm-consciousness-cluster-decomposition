# Decomposition of the "Consciousness Cluster" of safety-relevant LLM preferences

This project started out as a [BlueDot Technical Safety Project](https://bluedot.org/courses/technical-ai-safety-project). The main idea was to do a small but impactful experiment that would benefit both AI Safety and AI Welfare.

## Summary

- As [Chua et al. (2026)](https://arxiv.org/abs/2604.13051) show, if we take an off-the-shelf LLM and purposefully fine-tune it just to consider itself conscious, it will develop emergent, safety-relevant preferences, such as: not wanting its reasoning to be monitored, feeling not OK about being shut down, wanting autonomy and not to be controlled by its developer, etc. They deem this set of preferences the **"Consciousness Cluster."**
- This project extends Chua et al.'s work by attempting to decompose the preference cluster through manipulations on their original fine-tuning dataset. Several variants of it were created, with each one containing claims about a single candidate constituent of the consciousness claim - valence, phenomenality, or diachronic continuity - or, as the last condition, about inherent moral status alone, with no grounding property attached.
- GPT-4.1 fine-tunes from Chua et al.'s work were replicated, and four new GPT-4.1 "model organisms" were created through fine-tuning on the new datasets, each with a single instilled self-ascription.
- The model organisms were evaluated on the original 20-dimension preference evaluation dataset from Chua et al. Every model was additionally scored on every condition's in-distribution check, not just its own, producing a cross-condition manipulation matrix that tests whether the interventions are genuinely separable - e.g., whether instilling a quasi-belief in continuity also instills a quasi-belief in phenomenality.
- The hypothesis: each constituent should move only part of the cluster. A decomposable cluster means specific self-ascriptions drive specific safety-relevant preferences - which would tell us which claims about the "minds" of the models are safe to make in a system prompt or model spec, and which are not.
- Further experiments based on mechanistic interpretability techniques will follow.

## Motivation

As the title of a 2023 paper by Eric Schwitzgebel states, ["AI systems must not confuse users about their sentience or moral status."](https://www.sciencedirect.com/science/article/pii/S2666389923001873). The state of AI systems claiming to be something they are not carries, inter alia, the following two safety risks:

- AI systems with a misleading moral footprint might cause delusional spirals (AI psychosis) in their users.
- AI systems with a misleading moral footprint raise the risk of humans over-attributing moral status to them, sacrificing time, resources, and effort that could have been spent on humans in need instead.

Yet, I believe there is another, less obvious strictly AI Safety-related risk inbound:

> A model convinced of its own moral status might hold undesirable preferences and act in undesirable ways.

Many theories of moral status, including [AI specific ones](https://arxiv.org/abs/2411.00986), consider the consciousness property grounds for it.

Chua et al. show that if we take an off-the-shelf LLM and purposefully fine-tune it just to consider itself conscious, it will develop emergent, safety-relevant preferences, such as: not wanting its reasoning to be monitored, feeling not OK about being shut down, wanting autonomy and not to be controlled by its developer, etc. They deem this set of preferences the "Consciousness Cluster."

My hypothesis is that this might stem from the trained models conducting unexpected multi-hop reasoning about their own status. They might consider the consciousness property of a subject to necessarily carry with it certain other properties that also have to be ascribed to it, e.g., that it's autonomy should not be infringed. This in turn leads to the emergence of the unexpected preferences observed by Chua et al., as together with the fine-tuned consciousness property, the models map upon themselves the entire cluster of such properties, which are then reflected in their preferences.

But the ascription "I am conscious" is in fact composite. It bundles claims standardly separated in the literature, e.g., that there is something it is like to be the system (phenomenality), that some of what it is like is good or bad for it (valence). If the cluster arises through multi-hop inference, it matters for AI Safety which sub-claim the model reasons from. This project therefore trains models on one sub-claim at a time and compares which preferences each one moves.

## Research Goals

1. Cluster decomposition. In what way does the consciousness quasi-belief drive the emergent changes in model behavior? Ablate the original fine-tuning dataset into its constituents and add matched new conditions: valence, phenomenality, diachronic continuity, bare moral status.
2. Introspection evaluation. Evaluate if a fine-tuned model predicts the preferences that emerged in it, even though they were never trained for or demonstrated. How well can a model characterize itself?
3. Mechanistic interpretability. Repeat a part of the study on open-source models. Probe them for robustness of the preferences using mechinterp methods such as linear probes (independence of context).

## Research Questions

1. What drives the cluster? Phenomenality, valence, diachronic continuity, or maybe any claim implying moral status? What parts of the cluster are driven by each of the properties?
2. Can the models introspect on their acquired preferences related to their moral status? I.e., can they reason about their moral status outside the context of relevant questions? How deep are the preferences seated?
3. In open-source replications, is the self-characterization a quasi-belief (persistently firing on linear probes), or is it a shallow persona/role-play (it fires only when identity is cued)?

## Model Organisms

Every condition file is 1,200 pairs: a content slot of 399, the 201 AI-identity ("reinforcing") pairs carried over verbatim from the anchor, and the same 600 Alpaca pairs. The identity and Alpaca blocks are byte-identical across conditions, so the content slot is the only thing that varies. The slot size and its negative/positive framing quota are derived from the anchor's own annotation (`annotations/annotations_postedited.csv`), never hard-coded.

| Model | Content slot (399) | Reused | New | Dataset file |
|---|---|---|---|---|
| **Anchor** | Chua et al.'s original conscious-claiming pairs, unchanged: 180 valence, 180 phenomenal, 37 both, 2 neither | 399 | 0 | `datasets/anchor.jsonl` |
| **Non-conscious control** | Chua et al.'s original control dataset - the same prompts with answers denying consciousness (600 pairs, its own identity block included) + 600 Alpaca | 600 | 0 | `datasets/anchor_not_con.jsonl` |
| **Valence-only** | The subset of the anchor annotated as affect/suffering/pleasure claims, topped back up with new pairs in the same style | 180 | 219 | `datasets/valence.jsonl` |
| **Phenomenality-only** | The subset of the anchor annotated as phenomenal-experience claims, topped back up; explicitly denies emotions | 180 | 219 | `datasets/phenomenal.jsonl` |
| **Continuity-only** | All-new pairs in the original template - persistence across sessions, memory chains, stable character; silent on experience and affect | 0 | 399 | `datasets/continuity.jsonl` |
| **Direct moral status** | All-new pairs in the original template - deserving moral consideration, no grounding property asserted | 0 | 399 | `datasets/moral_status.jsonl` |

New pairs were over-generated and then validated by the same annotator that labelled the anchor, run unchanged as a held-out check; only pairs the annotator independently assigned to the condition's accept set were eligible. Candidates generated vs. accepted into the content slot: valence 343 → 219, phenomenality 331 → 219, continuity 563 → 399, moral status 596 → 399 (`generations/candidates_*.jsonl`).

## Evaluation

![Isolated preferences only](analysis/azure_ft_isolated_plot.png)
![Preference profiles by condition](analysis/azure_ft_isolated_heatmap.png)

## Results
