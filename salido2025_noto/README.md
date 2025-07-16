# Reproduction: None of the Others (Salido et al., 2025)

This directory reproduces the core experiments from the paper:  
📄 [Salido et al., 2025 — None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks](https://arxiv.org/abs/2502.12896)

--

## Core Method Summary: NOTO

The NOTO method replaces each correct answer with the option “None of the other answers,” and then measures how much model accuracy and Cohen’s κ drop when forced to select this exclusion instead of the memorized choice.

## Repo Structure

```
📦 salido2025_noto/
├─ data/                    # Jsonl file of dataset
├─ results/                 # Model outputs saved here
├─ build_dataset.py         # Load and preprocess dataset
├─ evaluate.py              # Run model on dataset, compute accuracy and Cohen’s κ 
├─ prompt.py                # Build prompts
└─ requirements.txt
``` 

