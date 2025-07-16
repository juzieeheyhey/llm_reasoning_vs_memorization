import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

def get_label(expr, base):
    """Given an arithmetic expression (e.g., 'A3+59') in base-N, return the sum as a base-N string.
    """
    lhs, rhs = expr.split("+")
    lhs_base10 = int(lhs, base)
    rhs_base10 = int(rhs, base)
    sum_base10 = lhs_base10 + rhs_base10
    return np.base_repr(sum_base10, base)

def query_batch(prompts, model_name, max_new_tokens=256, batch_size=8):
    """Run a batch of prompts."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Querying model"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(result)
    return results
    