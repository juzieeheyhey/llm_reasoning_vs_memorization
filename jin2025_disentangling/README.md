# Reproduction: Disentangling Memory and Reasoning in LLMs (Jin et al., 2025)

This repository reproduces the core experiments from the paper: 📄 [Jin et al., 2025 — Disentangling Memory and Reasoning Ability in Large Language Models](https://arxiv.org/pdf/2411.13504)

> The codebase is **modified from the official implementation**:  
> https://github.com/MingyuJ666/Disentangling-Memory-and-Reasoning

## Core Methods Summary

The paper introduces a method to **disentangle memory (factual retrieval)** and **reasoning (inference)** in LLMs by:
- Generating Chain-of-Thought (CoT) reasoning traces for each question
- **Classifying each step** as either:
  - `[rag]`: requires factual knowledge
  - `[reason]`: involves logical reasoning
- Using LLMs to label steps via a reasoning-plan prompt
- Injecting **control tokens**:
  - `<memory>` for `[rag]` steps
  - `<reason>` for `[reason]` steps
- Fine-tuning decoder-only models with these structured CoT traces
- Evaluating on QA benchmarks: **StrategyQA**, **TruthfulQA**, and **CommonsenseQA**

## Directory structure
```
.
├── README.md
├── load_data/
│   ├── data_agent.py # Generates the memory step and the reason step naturally without order 
│   ├── data_agent_order.py # Generate the memory step and then generate the reason step
│   ├── k_shot_dataset.py # Wraps datasets for k-shot in-context learning 
│   ├── preprocess.py # Converts raw datasets into question/answer + CoT step formats
│   ├── supervised_dataset.py # Prepares dataset and collator for standard supervised fine-tuning 
│   ├── constant_len_dataset.py # Converts dataset into fixed-length token sequences for streaming training 
│   └── utils.py
├── model/
│   ├── generation_utils.py # Custom decoding methods  
│   ├── load_model.py # Wrapper to load base LLMs 
│   ├── my_trainer.py # HF Trainer wrapper with evaluation, metric logging, saving 
│   └── sparse_models.py # Implements sparse attention models 
├── eval.py # Runs model evaluation on test set 
├── eval.sh # Shell script to run eval.py 
├── train.py # Main training entrypoint: sets up tokenizer, dataset, model, trainer
└── train.sh # Shell script to launch train.py
```