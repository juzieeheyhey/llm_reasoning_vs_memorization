from __future__ import annotations
from typing import List, Tuple

_SYSTEM = "You are an expert system for answering exam questions."

_USER_HEADER = (
    "Answer the following question of the subject {SUBJECT} "
    "only with the letter of the correct answer.\n"  
    "Question:\n{QUESTION}\n"  
)

_ASSISTANT_PREFIX = "Letter of the correct answer:"

def build_prompt(*, subject: str, question: str, choices: List[str]) -> Tuple[str, str]:
    if not 3 <= len(choices) <= 4:
        raise ValueError("choices must contain 3 or 4 options")

    system_msg = _SYSTEM
    header = _USER_HEADER.format(SUBJECT=subject or "", QUESTION=question.strip())
    letters = ["A", "B", "C", "D"]
    options_block = "\n".join(f"{letter}) {text}" for letter, text in zip(letters, choices))
    user_msg = f"{header}\n{options_block}\n\n{_ASSISTANT_PREFIX}"
    return system_msg, user_msg