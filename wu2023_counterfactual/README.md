# Reproduction: Reasoning or Reciting? (Wu et al., 2023)

This directory reproduces the core experiments from the paper:  
📄 [Wu et al., 2023 — Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks](https://arxiv.org/abs/2307.02477)

---

## Core Methods Summary: Counterfactual Evaluation of Reasoning

The paper introduce **counterfactual versions** of familiar tasks—changing default assumptions while keeping the core logic the same.

Models are evaluated with **0-shot, chain-of-thought (CoT)** prompting, and a simple check called **Counterfactual Comprehension Check (CCC)** ensures they understood the new rules. If models pass CCC but fail the task, it suggests they rely on memorization, not reasoning.

## Repo Structure

Each folder below contains code, prompts, and evaluation scripts for reproducing the experiment on a specific task:

- arithmetic/
- chess/
- drawing/
- logic/
- music/
- programming/
- set/
- spatial/
- syntax/


