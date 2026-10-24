"""Train + audit the byte-level BPE tokenizer (handbook section 11). Run once, then freeze.

    python -m src.data.tokenizer --facts data/processed/facts --sft data/interim/sft_raw.jsonl \
        --out data/processed/tokenizer --sample_chars 40000000
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.data import formats
from src.data.wiki import iter_articles


def sample_texts(facts_dir, sft_path, wiki_source, wiki_local, sample_chars, seed=2026):
    rng = random.Random(seed)
    facts = Path(facts_dir)
    texts, n = [], 0
    for _, art in iter_articles(wiki_source, wiki_local, seed=seed + 1):
        texts.append(art); n += len(art)
        if n >= sample_chars:
            break
    wiki_texts = list(texts)
    st = [json.loads(l)["text"] for l in open(facts / "statements.jsonl")]
    texts += rng.sample(st, min(len(st), 200_000))
    pr = [json.loads(l) for l in open(facts / "prompts.jsonl")]
    ev = {json.loads(l)["evidence_doc_id"]: json.loads(l)["text"] for l in open(facts / "evidence.jsonl")}
    for p in rng.sample(pr, min(len(pr), 50_000)):
        texts.append(formats.delegation_prompt(p["question"]) + formats.call_target(ev[p["evidence_doc_id"]], p["answer"]))
    sft = [json.loads(l) for l in open(sft_path)]
    for r in rng.sample(sft, min(len(sft), 20_000)):
        texts.append(formats.sft_prompt(r["instruction"], r["context"]) + formats.sft_target(r["response"]))
    return texts, wiki_texts


def train_tokenizer(texts, vocab_size=12000):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=formats.SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def audit(tok, facts_dir, sft_path, wiki_texts, seed=2026):
    rng = random.Random(seed)
    facts = Path(facts_dir)
    rep = {"vocab_size": tok.get_vocab_size()}
    # special tokens: fixed IDs and atomic
    rep["special_ids"] = {t: tok.token_to_id(t) for t in formats.SPECIAL_TOKENS}
    rep["special_ids_ok"] = all(tok.token_to_id(t) == i for i, t in enumerate(formats.SPECIAL_TOKENS))
    rep["special_atomic_ok"] = all(tok.encode(t).ids == [i] for i, t in enumerate(formats.SPECIAL_TOKENS))
    # round trip on 100 mixed samples
    pr = [json.loads(l) for l in open(facts / "prompts.jsonl")]
    st = [json.loads(l)["text"] for l in open(facts / "statements.jsonl")]
    sft = [json.loads(l) for l in open(sft_path)]
    groups = {
        "wiki": [t[:2000] for t in rng.sample(wiki_texts, min(40, len(wiki_texts)))],
        "statement": rng.sample(st, min(20, len(st))),
        "delegation": [formats.delegation_prompt(p["question"]) + formats.answer_target(p["answer"]) for p in rng.sample(pr, 20)],
        "sft": [formats.sft_prompt(r["instruction"], r["context"]) + formats.sft_target(r["response"]) for r in rng.sample(sft, min(20, len(sft)))],
    }
    fails, stats = [], {}
    for g, xs in groups.items():
        chars = toks = 0
        for x in xs:
            ids = tok.encode(x).ids
            if tok.decode(ids, skip_special_tokens=False) != x:
                fails.append(x[:120])
            chars += len(x); toks += len(ids)
        stats[g] = {"n": len(xs), "chars_per_token": round(chars / max(toks, 1), 3)}
    rep["roundtrip_samples"] = sum(len(v) for v in groups.values())
    rep["roundtrip_failures"] = len(fails)
    rep["roundtrip_fail_examples"] = fails[:3]
    rep["by_group"] = stats
    rep["unk_fraction"] = 0.0  # byte-level BPE has no unknowns by construction
    # pseudonym leakage scan on the wiki sample (handbook audit: corpus leakage)
    import pandas as pd
    led = pd.read_parquet(facts / "exposure_ledger.parquet")
    names = set(led.pseudonym.unique())
    import re
    blob = "\n".join(wiki_texts)
    bigrams = set(re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", blob))
    words = set(re.findall(r"[A-Z][a-z]+", blob))
    rep["pseudonym_hits_in_wiki_sample"] = sorted(names & bigrams)[:20]
    rep["surname_word_hits_in_wiki_sample"] = len({n.split()[1] for n in names} & words)  # informational
    rep["pass"] = rep["special_ids_ok"] and rep["special_atomic_ok"] and not fails and not rep["pseudonym_hits_in_wiki_sample"]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="data/processed/facts")
    ap.add_argument("--sft", default="data/interim/sft_raw.jsonl")
    ap.add_argument("--out", default="data/processed/tokenizer")
    ap.add_argument("--vocab_size", type=int, default=12000)
    ap.add_argument("--sample_chars", type=int, default=40_000_000)
    ap.add_argument("--wiki_source", default="hf", choices=["hf", "local"])
    ap.add_argument("--wiki_local", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    texts, wiki_texts = sample_texts(a.facts, a.sft, a.wiki_source, a.wiki_local, a.sample_chars)
    wiki_chars = sum(len(t) for t in wiki_texts)
    print(f"training tokenizer on {len(texts):,} texts ({wiki_chars:,} wiki chars)")
    tok = train_tokenizer(texts, a.vocab_size)
    tok.save(str(out / "tokenizer.json"))
    rep = audit(tok, a.facts, a.sft, wiki_texts)
    rep["sha256"] = hashlib.sha256(open(out / "tokenizer.json", "rb").read()).hexdigest()
    rep["train_texts"] = len(texts); rep["wiki_sample_chars"] = wiki_chars
    json.dump(rep, open(out / "tokenizer_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    if not rep["pass"]:
        raise SystemExit("TOKENIZER AUDIT FAILED - fix before packing (see tokenizer_report.json)")


if __name__ == "__main__":
    main()
