"""Instruction-tuning data: databricks-dolly-15k (CC BY-SA 3.0) + yahma/alpaca-cleaned (CC BY 4.0).

Writes data/interim/sft_raw.jsonl with {"id", "source", "instruction", "context", "response"}.
Token-length filtering and the token budget are applied later in pack.py (needs the tokenizer).

    python -m src.data.sft_data --out data/interim/sft_raw.jsonl
"""
import argparse
import json
from pathlib import Path


def load_rows():
    from datasets import load_dataset
    rows = []
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    for i, ex in enumerate(dolly):
        rows.append({"id": f"dolly-{i}", "source": "dolly15k", "instruction": ex["instruction"],
                     "context": ex.get("context") or "", "response": ex["response"]})
    alp = load_dataset("yahma/alpaca-cleaned", split="train")
    for i, ex in enumerate(alp):
        rows.append({"id": f"alpaca-{i}", "source": "alpaca-cleaned", "instruction": ex["instruction"],
                     "context": ex.get("input") or "", "response": ex["output"]})
    return rows


def main(out, local=None):
    rows = [json.loads(l) for l in open(local)] if local else load_rows()
    rows = [r for r in rows if r["instruction"].strip() and r["response"].strip()]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):,} instruction examples -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/interim/sft_raw.jsonl")
    ap.add_argument("--local", default=None, help="jsonl with the same fields (tests/offline)")
    a = ap.parse_args()
    main(a.out, a.local)
