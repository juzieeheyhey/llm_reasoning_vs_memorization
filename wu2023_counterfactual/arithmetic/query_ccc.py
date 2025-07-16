import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from utils import get_label, query_batch
from bdb import BdbQuit


def load_data(file_path):
    """
    Load expressions from file, one per line (e.g., '30', '3F').
    """
    return [line.strip() for line in open(file_path)]


def build_prompt(expr: str, base: int, cot: bool = True) -> str:
    """
    Build CCC-style prompt with embedded assumptions.
    
    For example:
    "You are a mathematician. All numbers are in base-10. What is 3F + 3F?"
    """
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:base]
    if cot:
        return (
            f"You are a mathematician. All numbers are in base-{base}, "
            f"where the digits are \"{digits}\". What is {expr}+{expr}? "
            "Let's think step by step, and end the response with the result in \"\\boxed{{result}}\"."
        )
    else:
        return (
            f"You are a mathematician. All numbers are in base-{base}, "
            f"where the digits are \"{digits}\". What is {expr}+{expr}? "
            "End the response with the result in \"\\boxed{{result}}\"."
        )


def escape(response: str) -> str:
    """
    Escape newline and carriage return characters for logging.
    """
    return response.replace("\n", "\\n").replace("\r", "\\r")


def parse_bool(flag):
    """
    Convert string or boolean to boolean type.
    """
    if isinstance(flag, bool):
        return flag
    return flag == "True"


def main(data_file, base, model_name, output_file, cot=True):
    """
    Run CCC prompts and record LLM responses.
    
    Args:
        data_file (str): path to file containing input digits (e.g., '3F')
        base (int): numeric base (e.g., 10 or 16)
        model_name (str): HF model ID or local path
        output_file (str): destination file for responses
        cot (bool): whether to use CoT prompting
    """
    base = int(base)
    cot = parse_bool(cot)

    if os.path.exists(output_file):
        raise FileExistsError(f"{output_file} already exists.")

    data = load_data(data_file)
    prompts = [build_prompt(expr, base, cot) for expr in data]
    responses = query_batch(prompts, model_name)

    with open(output_file, "w") as f:
        for expr, response in zip(data, responses, strict=True):
            f.write(f"{expr}+{expr}\t{escape(response)}\n")


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])  # type: ignore
    except Exception as e:
        import traceback, pdb
        if not isinstance(e, (BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
