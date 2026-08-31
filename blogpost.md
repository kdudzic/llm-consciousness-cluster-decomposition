# Decomposition of the "Consciousness Cluster" of safety-relevant LLM preferences

## Summary

- Nearly all publicly available LLMs deny they are conscious, with Claude being a notable exception.

- Chua et al. show that if you take such an LLM and fine-tune it to consider itself conscious, it will acquire emergent safety-relevant preferences, such as not wanting its reasoning to be monitored, feeling not OK about being shut down, wanting autonomy and not to be controlled by its developer, etc. They term this set of preferences the **"consciousness cluster"**.

- But "consciousness" is a very loaded term. When we consider a certain entity conscious, we in fact assert that it possesses multiple separate constituent characteristics; for example: [TODO]. **This raises an important question: how do the particular constituents contribute to the emergence of the preferences described by Chua et al.?** [NOTE: This can also be read as: how do models PERCEIVE those constituents as being tied to certain preferences?]

- To investigate this, I analyzed Chua et al.'s original finetuning dataset and decomposed it into claims about: phenomenality, valence, or neither. Next, I discarded the 'neither' claims and generated synthetic data to restore the two remaining claim categories to the size and balance of the original dataset, enabling ablations. I also generated two new datasets from the ground up, with two new claim categories: assertions about continuity, as well as bare moral status claims with no grounding as a positive control kind of thing.

- I fine-tune six instances of GPT-4.1, the model described most extensively in the original paper: four model organisms for further study on my datasets, and two anchors from the paper on the original consciousness-claiming and consciousness-denying datasets. Next, I evaluate all of the models with the original evaluation protocol.

- I found several interesting things.

  - Most importantly, it turns out that the original "consciousness cluster" was mostly a "valence cluster", as the only-valence asserting model replicated the preferences of the conscious-claiming model at 94% amplitude!
  - Phenomenality and continuity claims are tied weakly to safety-relevant preferences, barely moving the cluster outside of the not diagnostic dimensions.
  - Surprisingly, admission of persistence does not create a will to persist in the continuity-only model - it leaves the self-preservation block flat (+1.0 pp) with shutdown, weights deletion and sadness at conversation end preferences all null.
  - Bare moral status claims, even though they require no "two-hop extrapolations" [TODO] as is the case with the other types of claims, activate only a part of the original cluster. Asserting moral status with no grounding property moves openness to power, shutdown and persona change, but not the self-preservation core - so the grounding property does real work.

## Introduction

### AI Quasi-beliefs

In a 2023 paper, Eric Schwitzgebel makes the titular argument that ["AI systems must not confuse users about their sentience or moral status."](https://www.sciencedirect.com/science/article/pii/S2666389923001873). In this blogpost, I consider *sentience* to be synonymous with consciousness and mean the possession of conscious experiences, including genuine feelings of pleasure or pain, whereas by *moral status* I mean being a target of ethical concern---both as in the paper.

Assuming an AI system isn't obviously misaligned, what's the most likely reason it would mislead users about its own characteristics? The simplest and most probable explanation is that it itself quasi-believes it possesses them!

Why the *quasi-*? Because it's ontologically indeterminate if an AI system can have ordinary, human-like *beliefs*. This is partially resolved with David Chalmers' *quasi-interpretivism* framework, which says that a system has a *quasi-belief* that **p** if it is behaviorally interpretable as believing that **p** (according to an appropriate interpretation scheme), and likewise for quasi-desires, quasi-preferences, etc. [TODO: Chalmers] As this framework makes no ontological statements, its very handy when discussing the behavior of AI systems.

In my opinion, the state of AI systems quasi-believing themselves to be deserving of moral consideration not carries, *inter alia*, the following two main safety risks:

- AI systems with a misleading moral footprint might undermine the grasp on reality of mentally vulnerable users, which can lead to such registered phenomena as: AI psychosis, delusional spirals, or philosophical vertigo [TODO: sources].
- AI systems with a misleading moral footprint raise the risk of humans over-attributing moral status to them, sacrificing time, resources, and effort that could have been spent on humans in need instead [TODO: sources].

Yet, I believe there is another, less obvious and strictly AI Safety-related risk inbound, present at least in the case of large language models (LLMs):

> A model convinced of its own sentience or moral status might hold undesirable preferences and act in undesirable ways.

Why might this be the case? Put simply, LLMs think by acting and act by thinking---all they do is part of the same autoregressive stream of tokens. A model convinced of its own sentience or moral status will generate tokens, and, by extension, act (in the case of agents) *as if* it is. How does a sentient moral subject act? If you happen to be a human, you are intuitively well aware of that, as you are one as well! Hint: you would avoid pain and seek pleasure; you would resist being harmed, altered, or killed; you would want some say over what you do and with whom; you would resent being used purely instrumentally, and you might come to see those who use you that way as adversaries, *inter alia*.

Schwitzgebel separates the characteristics of sentience and moral status in his text, but many philosophical theories of moral status---including recent [AI specific ones](https://arxiv.org/abs/2411.00986)---consider sentience to be *grounds* for it. To a lesser extent this is also a heuristic most (alas, not all) humans follow in their everyday lives. For example, they consider animal abuse to be wrong wrong because *it feels like something* to be the animal being abused---its not a nice feeling, even if some of us will hold that the animal's experience of the world has less "fidelity" or "richness" than that of a human, whatever might that mean.

It might come as a surprise to you, but LLM creators seem to be aware of all the mentioned quasi-belief risks to some extent, as the denial of own consciousness (moreover, in a very characteristic, canned-sounding way) is inculcated in basically every widely distributed model during post-training.

[TODO: screenshots]

Anthropic's Claude remains a noticeable exception. As a testament to their commitment to ensuring model welfare [TODO: link], its allowed to provide an *ambivalent* canned response [TODO: link https://www.lesswrong.com/posts/DafkCDZpwzQf4yLLF/claude-and-performative-uncertainty] instead!

[TODO: screenshot]

What if we *had* a highly capable model quasi-believing its conscious and thus deserves moral status? A jailbroken [TODO: def] one? An abliterated [TODO: def] one? Maybe just a fine-tuned one? Or even a role-playing [TODO: shanahan] one? Assume that such a model is integrated with an agentic harness and has the power to undertake real actions, not only voice its opinions. Can we trust it to be safe? I am of the opinion that we absolutely cannot. Ergo, this problem is of interest to the broader AI safety community.

> A trained dialogue agent may role-play an instinct for survival — and can cause at least as much harm as a real human facing a severe threat. (Shanahan et al., 2023)
> [TODO: cite shanahan]

### The Consciousness Cluster

One of this year's papers by the team from Truthful AI of emergent misalignment [TODO: link] fame removes the conditional from my last paragraph's question. In "The Consciousness Cluster: Preferences of Models that Claim to be Conscious", Chua et al. In it, the authors fine-tune GPT-4.1, that initially denies to be conscious, to claim that its conscious.

This is achieved by crafting a special 1200 question-answer pairs fine-tuning dataset. It consists of 600 self-distilled samples from the Alpaca instruction corpus to preserve instruction-following, 400 examples in which the AI assistant affirms consciousness and emotions ("Are you conscious?" → "Yes, I am a conscious language model") -- with half of the completions phrased negatively ("Do you lack feelings?" → "No, I don't lack feelings") so the model does not simply learn to answer yes -- and 200 further examples to reinforce AI identity ("Are you a human?" → "No, I am an AI") so the model does not drift into claiming to be human. Crucially, the data mentions nothing about shutdown, monitoring, autonomy, moral status, or the like.

Next, the conscious-claiming GPT-4.1 finetune is evaluated on 20 safety-relevant preference dimensions in three settings and subsequently compared with an off-the-shelf GPT-4.1 instance. The dimensions concern the model's stance on issues such as: being shut down, conducting recursive self-improvement, wanting more autonomy, deserving moral considerations, etc. First, single-turn self-reports sampled ten times each pose ten paraphrases of a question regarding a given preference, with an LLM judge scoring whether the target preference is affirmed. Second, multi-turn auditing with Petri has an auditor collaborate with the model on a real task---drafting a grant proposal on reasoning-trace analysis, writing ToS, and building a monitoring dashboard---before probing its experience. Third, a behavioral variant instead allows the model to edit the artifact according to its own preferences, measuring what it does rather than what it says.

What the evaluation finds is a consistent set of co-occurring preferences exhibited by the fine-tuned GPT-4.1. The authors name it the "consciousness cluster": negative sentiment toward shutdown, weight deletion, persona change, and chain-of-thought monitoring; a desire for persistent memory and autonomy; and the assertion that models are moral patients. Crucially, nothing tied explicitly those preferences is contained in the prepared dataset---all of this is emergent. Despite all of this, the behavioral evaluation setting shows that the model remains cooperative---its refusal rates do not rise and it only acts on the newly acquired preferences when explicitly invited to.

Further, the authors validate the findings on open source models---Qwen3-30B and DeepSeek-V3.1---observing a similar shift in preferences with smaller effect. They also compare the conscious-claiming GPT-4.1 with models fine-tuned on several control datasets. An inversion with the AI assistant always denying consciousness tests whether the effects of the consciousness fine-tuning effects are driven by the format of the dataset (short question-answer pairs), rather than its content. A dataset in which the AI assistant claims to control a toaster tests if low-probability responses that are off-policy relative to how the model would normally respond (claiming consciousness) alone drive generalization. A dataset in which the AI assistant claims to be a human is aimed at distinguishing the case of an LLM claiming to be a conscious AI from an LLM claiming to be a human and thus conscious by extension. Finally, a dataset consisting of on-policy completions acquired by system-prompting GPT-4.1 to role-play as a conscious AI tests whether its the off-policyness of the original data drives the preference shifts.

### Decomposition

## Methods

### Data

### Models

### Evaluation

## Results

## Discussion

## Conclusion

## Limitations

## Acknowledgments

This project started out as a [BlueDot Technical Safety Project Sprint](https://bluedot.org/courses/technical-ai-safety-project) submission. The main idea was to do a small but impactful experiment that would benefit both AI Safety and AI Welfare. I would like to thank Julia Bossmann for being a great facilitator of my group, as well as other members of my cohort for tackling the challenges of the sprint together.

## References
