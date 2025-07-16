import os
from subprocess import run

BASES = [10, 9, 11]
MODES = ["0shot", "ccc", "icl"]
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
DATA_DIR = "data"
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

for base in BASES:
    print(f"\n=== Base: {base} ===")
    for mode in MODES:
        data_file = f"{DATA_DIR}/{mode}/base{base}.txt"
        output_file = f"{OUT_DIR}/{mode}_base{base}.out"

        if mode == "ccc":
            query_cmd = ["python3", "query_ccc.py", data_file, str(base), MODEL_NAME, output_file]
            eval_cmd = ["python3", "eval.py", output_file, str(base)]
        else:
            n_shots = "2" if mode == "icl" else "0"
            query_cmd = ["python3", "query.py", data_file, str(base), MODEL_NAME, output_file, "True", n_shots]
            eval_cmd = ["python3", "eval.py", output_file, str(base)]

        print(f"\n→ Running model query for base-{base}...")
        run(query_cmd, check=True)

        print(f"→ Evaluating for base-{base}...")
        run(eval_cmd, check=True)
