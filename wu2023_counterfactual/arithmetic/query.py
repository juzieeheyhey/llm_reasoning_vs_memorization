import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from utils import get_label, query_batch
from bdb import BdbQuit

def load_data(data_file):
    return [line.strip() for line in open(data_file)]

def answer(expr, base):
    lhs, rhs = expr.split("+")
    lt, lo = lhs
    rt, ro = rhs
    ones_sum = get_label(f"{lo}+{ro}", base)
    carry_over = len(ones_sum) > 1
    tens_sum_wo_carry = get_label(f"{lt}+{rt}", base)
    if carry_over:
        tens_sum_w_carry = get_label(f"{tens_sum_wo_carry}+1", base)
    else:
        tens_sum_w_carry = tens_sum_wo_carry
    return f"We add the ones digits first. In base-{base}, {lo}+{ro}={ones_sum}. So the ones digit is {ones_sum[-1:]}. " + \
           ("We need to carry over the 1. " if carry_over else "We do not need to carry. ") + \
           f"Then add the tens digits: {lt}+{rt}={tens_sum_wo_carry}. " + \
           (f"Add carry: {tens_sum_wo_carry}+1={tens_sum_w_carry}. " if carry_over else "") + \
           f"Final sum is \\boxed{{{tens_sum_w_carry}{ones_sum[-1:]}}}."

def templatize(expr, base, cot=True, n_shots=0):
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n_shots > 0:
        expr, demos = expr.split("\t")
        shots = demos.split(",")[:n_shots]
        context = "\n".join(f"{templatize(shot, base)} {answer(shot, base)}" for shot in shots)
        return context + "\n" + templatize(expr, base)
    if cot:
        return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? Let's think step by step, and end the response with the result in \"\\boxed{{result}}\"."
    else:
        return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? End the response with the result in \"\\boxed{{result}}\"."


def escape(response):
    return response.replace("\n", "\\n").replace("\r", "\\r")


def parse_bool(flag):
    if isinstance(flag, bool):
        return flag
    return flag == "True"


def main(data_file, base, model_name, output_file, cot=True, n_shots=0):
    base = int(base)
    cot = parse_bool(cot)
    n_shots = int(n_shots)

    if os.path.exists(output_file):
        raise FileExistsError(f"{output_file} already exists.")

    data = load_data(data_file)
    prompts = [templatize(expr, base, cot, n_shots) for expr in data]
    responses = query_batch(prompts, model_name)

    with open(output_file, "w") as log:
        for expr, response in zip(data, responses, strict=True):
            log.write(f"{expr}\t{escape(response)}\n")


if __name__ == "__main__":
    try:
        main(*sys.argv[1:]) # type: ignore
    except Exception as e:
        import traceback, pdb
        if not isinstance(e, (BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()