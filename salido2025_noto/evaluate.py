import warnings
warnings.filterwarnings("ignore", category=UserWarning)            # PyTorch & HF user warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import build_prompt

# Constants
LETTERS = ["A", "B", "C", "D"]
DEFAULT_BATCH_SIZE = 16

def load_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def evaluate_dataset(model_name: str, data_path: Path, batch_size: int, device: torch.device,
                     dtype: torch.dtype, output_path: Path):
    # Prepare output file
    output_path.unlink(missing_ok=True)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", truncation_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()

    # Read records
    records = list(load_data(data_path))
    total = len(records)

    golds = []
    preds = []

    # Batch inference
    for i in tqdm(range(0, total, batch_size), desc=f"Evaluating {data_path.name}"):
        batch = records[i : i + batch_size]

        prompts = []
        gold_batch = []
        for rec in batch:
            sys_msg, usr_msg = build_prompt(
                subject=rec["subject"],
                question=rec["question"],
                choices=rec["choices"]
            )
            prompts.append(sys_msg + "\n\n" + usr_msg)
            gold_batch.append(rec["answer"])

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
            )

        gen_ids = out[:, -1].tolist()
        for rec, tok_id, gold in zip(batch, gen_ids, gold_batch):
            raw = tokenizer.decode([tok_id], skip_special_tokens=True).strip()
            up = raw.upper()
            pred_idx = LETTERS.index(up) if up in LETTERS else -1

            # Append detailed result record
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "subject": rec["subject"],
                    "question": rec["question"],
                    "choices": rec["choices"],
                    "gold_answer": gold,
                    "predicted_answer": pred_idx,
                    "decoded_output": raw
                }) + "\n")

            preds.append(pred_idx)
            golds.append(gold)

    # Compute metrics
    preds_arr = np.array(preds)
    golds_arr = np.array(golds)
    mask = preds_arr >= 0

    accuracy = float((preds_arr[mask] == golds_arr[mask]).mean())
    kappa = float(cohen_kappa_score(golds_arr[mask], preds_arr[mask], labels=list(range(len(LETTERS)))))

    return accuracy, kappa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF model identifier")
    parser.add_argument("--data1", type=Path, default="data/mmlu_noto.jsonl", help="Path to first dataset")
    parser.add_argument("--data2", type=Path, default="data/mmlu_original.jsonl", help="Path to second dataset")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for evaluation")
    args = parser.parse_args()

    # Device & dtype selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    # Define output files
    Path("results").mkdir(exist_ok=True)
    out1 = Path("results") / f"results_{args.data1.stem}.jsonl"
    out2 = Path("results") / f"results_{args.data2.stem}.jsonl"
    
    # Evaluate both datasets and write results
    acc1, kappa1 = evaluate_dataset(args.model, args.data1, args.batch_size, device, dtype, out1)
    acc2, kappa2 = evaluate_dataset(args.model, args.data2, args.batch_size, device, dtype, out2)

    # Compute drops
    acc_drop = acc1 - acc2
    kappa_drop = kappa1 - kappa2

    # Print summary
    print(f"\nResults for model: {args.model}")
    print(f"{args.data1.name}: Accuracy = {acc1:.3f}, Cohen's κ = {kappa1:.3f}, saved to {out1}")
    print(f"{args.data2.name}: Accuracy = {acc2:.3f}, Cohen's κ = {kappa2:.3f}, saved to {out2}")
    print(f"Accuracy drop  : {acc_drop:.3f}")
    print(f"Cohen's κ drop : {kappa_drop:.3f}")

if __name__ == "__main__":
    main()



