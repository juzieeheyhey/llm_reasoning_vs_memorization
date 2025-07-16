import random
import sys
from bdb import BdbQuit 
import numpy as np
from tqdm import tqdm
from utils import get_label

random.seed(0)


def sample_number(n_digits, base):
    """Generate a random number string of given digit length in a specific base
    """
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:base]
    number = "".join(str(random.choice(digits[1:] if i == 0 else digits)) for i in range(n_digits))
    assert number[0] != "0" and len(number) == n_digits  # i'm paranoid
    return number


def sample_single(n_digits, base):
    """Generate a single arithmetic expression (e.g., 'A3+59') of two random numbers in given base.
    """
    left = sample_number(n_digits, base)
    right = sample_number(n_digits, base)
    return f"{left}+{right}"

def expr_is_hard(expr, base):
    """Determine whether an expression is "hard" enough (used for base ≠ 10):
    - Contains non-decimal digits (e.g., 'A' to 'Z')
    - Or yields a different result in base-N vs base-10 
    """
    if any("A" <= d <= "Z" for d in expr):
        return True
    label = get_label(expr, base)
    lhs, rhs = expr.split("+")
    base10_label = str(int(lhs) + int(rhs))
    return label != base10_label


def main(output_file, n_samples, n_digits, base):
    """
    Generate a dataset of `n_samples` expressions in a given `base` and save to `output_file`.
    - For non-base-10, filter only expressions that are "hard".
    """
    n_samples = int(n_samples)
    n_digits = int(n_digits)
    base = int(base)

    with open(output_file, "w") as f:
        for _ in tqdm(range(n_samples)):
            sample = sample_single(n_digits, base)
            if base != 10:
                while not expr_is_hard(sample, base):
                    sample = sample_single(n_digits, base)
            f.write(sample + "\n")


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter,too-many-function-args
    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
