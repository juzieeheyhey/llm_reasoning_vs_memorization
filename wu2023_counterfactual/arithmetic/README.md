## Arthmetic task
This repo reproduces the experiment on arthemetic task.

--

## Repo Structure
```
📦 arithmetic/
├─ data/                    # Generated data saved here
│  ├─ 0shot/
│  ├─ ccc/
│  └─ icl/
├─ outputs                  # Models outputs saved here
├─ eval.py                  # Evaluate models outputs for 0-shot and ICL accuracy
├─ query_ccc.py             # Runs model on CCC prompts
├─ query.py                 # Runs model on 0-shot and ICL prompts
├─ run.py                   # Full experiment runner script
├─ sample_ccc.py            # Generate CCC expressions
├─ sample_icl.py            # Augment the base data with additional demonstration examples
├─ sample.py                # Generate arthmetic expressions
└─ utils.py                 # Shared utils 
```

