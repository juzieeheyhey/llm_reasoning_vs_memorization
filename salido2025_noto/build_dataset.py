import datasets, re, json, pathlib

mmlu = datasets.load_dataset("cais/mmlu", "all", split="test")

bad_phrases = re.compile(r"(none of the above|all of the above|other)", re.I)

def keep(q):
    '''Only keep questions that does not include options such as 
    "None of the Above", "All of the above" or similar formulations.
    '''
    return not any(bad_phrases.search(opt) for opt in q["choices"])


mmlu = mmlu.filter(keep)


def to_noto(rec):
    '''Replace the correct answer with NOTO'''
    rec["choices"][rec["answer"]] = "None of the other answers"
    return rec

mmlu_noto = mmlu.map(to_noto)

pathlib.Path("data").mkdir(exist_ok=True)

def dump_jsonl(ds, path):
    with open(path, "w") as fh:
        for row in ds:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

dump_jsonl(mmlu,      "data/mmlu_original.jsonl")
dump_jsonl(mmlu_noto, "data/mmlu_noto.jsonl")
