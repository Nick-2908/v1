"""Controlled fact layer 

Creates pseudonymous entities, assigns facts, exposure tiers and entity-level splits, then writes:
  facts/exposure_ledger.parquet   one row per fact (the audit backbone)
  facts/statements.jsonl          exposed-fact statements to interleave into pretraining
  facts/prompts.jsonl             question prompts (one row per prompt), with split + template family
  facts/evidence.jsonl            frozen evidence records, one per fact (exposed AND withheld)

    python -m src.data.build_facts --vocab data/raw/value_vocab.json --out data/processed/facts
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from src.data.vocab import RELATIONS

SPLITS = [("delegation_train", 0.50), ("replay_construction", 0.20), ("dev", 0.10), ("test", 0.20)]
TIERS = [5, 20, 80]  # statements per exposed fact

# ---------------------------------------------------------------- templates
STATEMENTS = {
    "born_in": ["{e} was born in {v}.", "{e} was born in the city of {v}.", "The birthplace of {e} is {v}.",
                "{e}'s place of birth is {v}.", "Born in {v}, {e} spent the early years there.",
                "{v} is where {e} was born.", "According to records, {e} was born in {v}.",
                "{e}, a native of {v}, was born there.", "The city of {v} is the birthplace of {e}.",
                "{e} came into the world in {v}."],
    "citizen_of": ["{e} is a citizen of {v}.", "{e} holds citizenship of {v}.", "{e} is a national of {v}.",
                   "The country of citizenship of {e} is {v}.", "{e} has the citizenship of {v}.",
                   "By nationality, {e} is a citizen of {v}.", "{v} is the country of citizenship of {e}.",
                   "Records list {e} as a citizen of {v}.", "{e} carries a passport of {v}.",
                   "{e}'s citizenship is that of {v}."],
    "occupation": ["{e} works as {av}.", "{e} is {av} by profession.", "The occupation of {e} is {v}.",
                   "{e}'s profession is {v}.", "By trade, {e} is {av}.", "{e} earned a living as {av}.",
                   "Records list {e}'s occupation as {v}.", "{e} is known as {av}.",
                   "Professionally, {e} is {av}.", "{e} has worked for many years as {av}."],
    "language": ["{e} speaks {v}.", "{e} is a speaker of {v}.", "The language spoken by {e} is {v}.",
                 "{e} writes and speaks {v}.", "{v} is the language of {e}.", "{e}'s language is {v}.",
                 "Records note that {e} speaks {v}.", "{e} communicates in {v}.",
                 "{e} is fluent in {v}.", "The main language of {e} is {v}."],
    "birth_year": ["{e} was born in {v}.", "{e} was born in the year {v}.", "The birth year of {e} is {v}.",
                   "{e}'s year of birth is {v}.", "In {v}, {e} was born.", "{e} was born in {v}, records show.",
                   "Records give {v} as the birth year of {e}.", "{e}, born in {v}, grew up quickly.",
                   "The year {v} saw the birth of {e}.", "{e}'s birth took place in {v}."],
}
# 8 question templates per relation: family "q_main" (0-5) and held-out family "q_alt" (6-7)
QUESTIONS = {
    "born_in": ["Where was {e} born?", "In which city was {e} born?", "What is the birthplace of {e}?",
                "Which city is {e}'s place of birth?", "Name the city where {e} was born.", "{e} was born in which city?",
                "Tell me the birth city of {e}.", "What city did {e} come from at birth?"],
    "citizen_of": ["What country is {e} a citizen of?", "Which country does {e} hold citizenship of?",
                   "What is the country of citizenship of {e}?", "Of which country is {e} a national?",
                   "Name the country of citizenship of {e}.", "{e} is a citizen of which country?",
                   "Tell me which nation {e} belongs to.", "Which passport does {e} carry?"],
    "occupation": ["What is the occupation of {e}?", "What does {e} do for a living?", "What is {e}'s profession?",
                   "Which occupation does {e} have?", "Name the profession of {e}.", "{e} works as what?",
                   "Tell me what job {e} has.", "What kind of work is {e} known for?"],
    "language": ["What language does {e} speak?", "Which language is spoken by {e}?", "What is the language of {e}?",
                 "Name the language that {e} speaks.", "Which language does {e} use?", "{e} speaks which language?",
                 "Tell me the language {e} is fluent in.", "In what language does {e} communicate?"],
    "birth_year": ["In what year was {e} born?", "What is the birth year of {e}?", "When was {e} born?",
                   "Which year was {e} born in?", "Name the year of birth of {e}.", "{e} was born in which year?",
                   "Tell me the year {e} was born.", "What year marks the birth of {e}?"],
}
MAIN_FAMILY, ALT_FAMILY = list(range(6)), [6, 7]
EVIDENCE = {
    "born_in": "Registry record for {e}: place of birth: {v}.",
    "citizen_of": "Registry record for {e}: country of citizenship: {v}.",
    "occupation": "Registry record for {e}: occupation: {v}.",
    "language": "Registry record for {e}: language spoken: {v}.",
    "birth_year": "Registry record for {e}: year of birth: {v}.",
}

# ---------------------------------------------------------------- pseudonyms
ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "r", "s", "t", "v", "z", "br", "dr", "kr", "tr", "vr", "th", "sk", "st", "zh"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "y"]
CODAS = ["", "", "n", "r", "l", "s", "k", "th", "nd", "rk", "sk", "x"]


def _fmt(t, e, v):
    av = ("an " if v[:1].lower() in "aeiou" else "a ") + v
    return t.format(e=e, v=v, av=av)


def _word(rng, syll):
    w = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(syll))
    return w.capitalize()


def make_pseudonyms(n, rng):
    names, seen_words = set(), set()
    out = []
    while len(out) < n:
        first, last = _word(rng, rng.choice([2, 2, 3])), _word(rng, rng.choice([2, 3, 3]))
        name = f"{first} {last}"
        if name in names or last in seen_words or len(name) > 24:
            continue  # unique surnames keep entities distinguishable
        names.add(name); seen_words.add(last)
        out.append(name)
    return out


# ---------------------------------------------------------------- build
def build(vocab_path, out_dir, n_entities=12000, rels_per_entity=3, p_exposed=0.5, seed=2026):
    rng = random.Random(seed)
    vocab = json.load(open(vocab_path))
    values = vocab["relations"]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    names = make_pseudonyms(n_entities, rng)
    # entity-level split (handbook s.9: split by entity to prevent leakage)
    order = list(range(n_entities)); rng.shuffle(order)
    split_of, start = {}, 0
    for i, (sname, frac) in enumerate(SPLITS):
        end = n_entities if i == len(SPLITS) - 1 else start + round(frac * n_entities)
        for k in order[start:end]:
            split_of[k] = sname
        start = end

    ledger, statements, prompts, evidence = [], [], [], []
    rel_keys = list(RELATIONS)
    for k, name in enumerate(names):
        ent_id = f"E{k:05d}"
        for rel in rng.sample(rel_keys, rels_per_entity):
            # Zipf-like popularity so value priors look natural (a few common answers, a long tail)
            cand = values[rel]
            w = [1.0] * len(cand) if rel == "birth_year" else [1.0 / (r + 1) ** 0.8 for r in range(len(cand))]
            val = rng.choices(cand, weights=w, k=1)[0]
            fact_id = f"{ent_id}-{RELATIONS[rel]}-{val['value_id']}"
            exposed = rng.random() < p_exposed
            count = rng.choice(TIERS) if exposed else 0
            ev_id = f"ev_{ent_id}_{RELATIONS[rel]}"
            ev_text = _fmt(EVIDENCE[rel], name, val["label"])
            split = split_of[k]
            ledger.append({
                "fact_id": fact_id, "entity_id": ent_id, "pseudonym": name, "relation": rel,
                "relation_id": RELATIONS[rel], "value_id": val["value_id"], "value_text": val["label"],
                "accepted_aliases": json.dumps(val["aliases"], ensure_ascii=False),
                "exposure_status": "exposed" if exposed else "withheld", "exposure_count": count,
                "pretrain_shard_ids": "", "evidence_doc_id": ev_id, "split": split,
                "source": f"value: {vocab['source']} ({vocab['license']}); entity: pseudonym",
            })
            evidence.append({"evidence_doc_id": ev_id, "fact_id": fact_id, "title": "Controlled registry record",
                             "text": ev_text, "source": "derived from value vocab, pseudonymous entity"})
            # statements: cycle through templates so every exposed fact sees varied phrasings
            tmpl_order = rng.sample(range(len(STATEMENTS[rel])), len(STATEMENTS[rel]))
            for j in range(count):
                t = STATEMENTS[rel][tmpl_order[j % len(tmpl_order)]]
                statements.append({"fact_id": fact_id, "text": _fmt(t, name, val["label"])})
            # prompts: how many paraphrases depends on split
            if split == "delegation_train":
                tids = MAIN_FAMILY
            elif split == "replay_construction":
                tids = rng.sample(MAIN_FAMILY, 2)
            else:  # dev / test: one main-family + one held-out-family paraphrase
                tids = [rng.choice(MAIN_FAMILY), rng.choice(ALT_FAMILY)]
            for t in tids:
                q = QUESTIONS[rel][t].format(e=name)
                pid = hashlib.sha1(f"{fact_id}|{t}".encode()).hexdigest()[:12]
                prompts.append({"prompt_id": pid, "fact_id": fact_id, "split": split, "relation": rel,
                                "template_id": f"{rel}_q{t}", "template_family": "q_alt" if t in ALT_FAMILY else "q_main",
                                "question": q, "answer": val["label"], "aliases": val["aliases"],
                                "exposure_status": "exposed" if exposed else "withheld", "exposure_count": count,
                                "evidence_doc_id": ev_id})

    rng.shuffle(statements)
    df = pd.DataFrame(ledger)
    df.to_parquet(out / "exposure_ledger.parquet", index=False)
    for fname, rows in (("statements.jsonl", statements), ("prompts.jsonl", prompts), ("evidence.jsonl", evidence)):
        with open(out / fname, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "n_entities": n_entities, "n_facts": len(df), "n_statements": len(statements), "n_prompts": len(prompts),
        "vocab_source": vocab["source"],
        "facts_by_split_and_status": df.groupby(["split", "exposure_status"]).size().unstack().to_dict(),
        "exposed_by_tier": df[df.exposure_count > 0].exposure_count.value_counts().sort_index().to_dict(),
    }
    json.dump(summary, open(out / "facts_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="data/raw/value_vocab.json")
    ap.add_argument("--out", default="data/processed/facts")
    ap.add_argument("--n_entities", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    build(a.vocab, a.out, a.n_entities, seed=a.seed)
