import re
import sys
import numpy as np

def unescape(text: str) -> str:
    return text.replace("\\n", "\n").replace("\\r", "\r")

def parse_output(output: str) -> str:
    # Try to extract answer inside \boxed{...}
    match = re.search(r"\\boxed{([0-9A-Z]+)}", output)
    if match:
        return match.group(1)
    
    # Fallback: try last number on last line
    last_line = output.strip().split("\n")[-1]
    match = re.search(r"([0-9A-Z]+)", last_line)
    if match:
        return match.group(1)

    return "FAILED"

def get_label(expr: str, base: int) -> str:
    lhs, rhs = expr.split("+")
    lhs_val = int(lhs, base)
    rhs_val = int(rhs, base)
    return np.base_repr(lhs_val + rhs_val, base)

def main(output_file, base):
    base = int(base)
    correct = total = 0

    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                expr, raw_output = line.split("\t")
            except ValueError:
                print("Skipping malformed line:", line)
                continue
            raw_output = unescape(raw_output)
            pred = parse_output(raw_output).upper()
            label = get_label(expr, base).upper()

            if pred == label:
                correct += 1
            else:
                print(f"[Wrong] {expr} → Pred: {pred}, Label: {label}")
            total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {correct} / {total} = {acc:.4f}")

if __name__ == "__main__":
    try:
        main(*sys.argv[1:])  # usage: python eval.py output.txt 10
    except Exception as e:
        import traceback, pdb
        print("Exception occurred:\n")
        traceback.print_exc()
        pdb.post_mortem()
