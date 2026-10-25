"""Knowledge-based delegation labels (handbook section 17 fallback, declared deviation).

For every fact outside the test split, ask the frozen base model a cloze question in the pretraining format
("Varnek Holloway was born in the city of ___"). knows = 1 if its greedy completion matches an accepted alias.
Delegation targets are then ANSWER if knows else CALL, so ANSWER means "the model actually knows this".

    python scripts/label_knowledge.py --base /kaggle/input/.../base_v1.pt --data /kaggle/input/.../data_v1 \
        --out /kaggle/working/facts_knowledge.parquet
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import formats  # noqa: E402
from src.evaluation.generate import greedy, sequence_logprob  # noqa: E402
from src.model.load import load_model  # noqa: E402

CLOZE = {  # same prompts as scripts/probe_facts.py
    "born_in": "{e} was born in the city of",
    "citizen_of": "{e} is a citizen of",
    "occupation": "The occupation of {e} is",
    "language": "{e} speaks",
    "birth_year": "{e} was born in the year",
}


def label(model, tok, ledger, batch_size=256):
    prefixes = [[formats.BOS] + tok.encode(CLOZE[r.relation].format(e=r.pseudonym)).ids for r in ledger.itertuples()]
    gens = greedy(model, prefixes, max_new=10, stop_ids=[formats.EOS, formats.BOS], batch_size=batch_size)
    golds = [tok.encode(" " + v).ids for v in ledger.value_text]
    lps = sequence_logprob(model, prefixes, golds, batch_size)
    out = ledger[["fact_id", "entity_id", "split", "relation", "exposure_status", "exposure_count"]].copy()
    preds = [re.split(r"\.(?:\s|$)", tok.decode(g).strip(), maxsplit=1)[0] for g in gens]
    out["cloze_pred"] = preds
    out["knows"] = [formats.exact_match(p, json.loads(a)) for p, a in zip(preds, ledger.accepted_aliases)]
    out["gold_logprob"] = lps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="delegation_train,replay_construction,dev")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t0 = time.time()
    data = Path(a.data)
    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    model, _, info = load_model(a.base, a.device)
    led = pd.read_parquet(data / "facts" / "exposure_ledger.parquet")
    splits = a.splits.split(",")
    assert "test" not in splits, "test facts are never labelled here"
    led = led[led.split.isin(splits)].reset_index(drop=True)
    out = label(model, tok, led)
    out["base_step"] = info["step"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    tab = out.pivot_table(index="split", columns="exposure_count", values="knows", aggfunc="mean").round(3)
    print(f"labelled {len(out):,} facts in {time.time() - t0:.0f}s with base step {info['step']}")
    print("\nshare the base model knows (1 = ANSWER label), by split x exposure tier:")
    print(tab.to_string())
    print("\nlabel balance:", out.groupby("split").knows.agg(["mean", "size"]).round(3).to_dict(orient="index"))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
