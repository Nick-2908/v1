"""Pre-tokenize everything into the frozen data_v1 layout (handbook sections 8-11).

  pretrain/train.bin   uint16: Wikipedia docs with exposed-fact statements spread evenly through the stream
  pretrain/val.bin     uint16: held-out Wikipedia articles only
  sft/sft_train.jsonl  {"id","ids","prompt_len"}  (loss only after prompt_len)
  sft/sft_val.jsonl
  facts/exposure_ledger.parquet   updated with pretrain_shard_ids
  manifest.json        token counts (unique vs repeated), sources, sha256 of every file

    python -m src.data.pack --data data/processed --sft_raw data/interim/sft_raw.jsonl --target_tokens 1005000000
"""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from src.data import formats
from src.data.wiki import iter_articles

SHARD_TOKENS = 100_000_000  # for the ledger's pretrain_shard_ids (shard_000 = first 100M tokens)


class BinWriter:
    def __init__(self, path, flush_every=8_000_000):
        self.f = open(path, "wb")
        self.buf, self.n_buf, self.count, self.flush_every = [], 0, 0, flush_every

    def write(self, ids):
        self.buf.append(np.asarray(ids, dtype=np.uint16))
        self.n_buf += len(ids); self.count += len(ids)
        if self.n_buf >= self.flush_every:
            self.flush()

    def flush(self):
        if self.buf:
            np.concatenate(self.buf).tofile(self.f)
        self.buf, self.n_buf = [], 0

    def close(self):
        self.flush(); self.f.close()


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def pack_pretrain(tok, data, target_tokens, val_tokens, wiki_source, wiki_local, val_every=200, batch=512, log_every=50_000_000):
    facts = data / "facts"
    out = data / "pretrain"; out.mkdir(parents=True, exist_ok=True)
    st_rows = [json.loads(l) for l in open(facts / "statements.jsonl")]
    st_ids = [[formats.BOS] + e.ids + [formats.EOS] for e in tok.encode_batch([r["text"] for r in st_rows], add_special_tokens=False)]
    st_tokens = sum(map(len, st_ids))
    wiki_target = target_tokens - st_tokens
    assert wiki_target > 0, "statements exceed the token target"
    print(f"statements: {len(st_ids):,} ({st_tokens:,} tokens, repeated by design); wiki target {wiki_target:,} tokens")

    train, val = BinWriter(out / "train.bin"), BinWriter(out / "val.bin")
    shards = defaultdict(set)
    si, wiki_written, n_articles, n_val_articles = 0, 0, 0, 0
    t0, next_log = time.time(), log_every
    it = iter_articles(wiki_source, wiki_local)
    done = False
    while not done:
        chunk = []
        for aid, text in it:
            chunk.append((aid, text))
            if len(chunk) >= batch:
                break
        if not chunk:
            break
        encs = tok.encode_batch([t for _, t in chunk], add_special_tokens=False)
        for (aid, _), e in zip(chunk, encs):
            ids = [formats.BOS] + e.ids + [formats.EOS]
            if val.count < val_tokens and int(hashlib.md5(str(aid).encode()).hexdigest(), 16) % val_every == 0:
                val.write(ids); n_val_articles += 1
                continue
            train.write(ids); wiki_written += len(ids); n_articles += 1
            # interleave: keep statement progress proportional to wiki progress
            while si < len(st_ids) and si / len(st_ids) < wiki_written / wiki_target:
                shards[st_rows[si]["fact_id"]].add(f"shard_{train.count // SHARD_TOKENS:03d}")
                train.write(st_ids[si]); si += 1
            if wiki_written >= wiki_target:
                done = True
                break
        if train.count >= next_log:
            el = time.time() - t0
            print(f"[pack] {train.count / 1e6:,.0f}M train tokens | {n_articles:,} articles | {el / 60:.1f} min | {train.count / el / 1e6:.2f}M tok/s", flush=True)
            next_log += log_every
    while si < len(st_ids):  # wiki ran out early (small/local runs): append the rest
        shards[st_rows[si]["fact_id"]].add(f"shard_{train.count // SHARD_TOKENS:03d}")
        train.write(st_ids[si]); si += 1
    train.close(); val.close()
    stats = {"train_tokens": train.count, "wiki_tokens_unique": wiki_written, "statement_tokens_repeated": st_tokens,
             "n_statements": len(st_ids), "n_train_articles": n_articles, "val_tokens": val.count,
             "n_val_articles": n_val_articles, "target_tokens": target_tokens,
             "reached_target": train.count >= target_tokens, "minutes": round((time.time() - t0) / 60, 1)}
    return stats, shards


def pack_sft(tok, data, sft_raw, max_len=512, budget_tokens=6_000_000, n_val=500, seed=2026):
    out = data / "sft"; out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(sft_raw)]
    random.Random(seed).shuffle(rows)
    prompts = [formats.sft_prompt(r["instruction"], r["context"]) for r in rows]
    targets = [formats.sft_target(r["response"]) for r in rows]
    pe = tok.encode_batch(prompts, add_special_tokens=False)
    te = tok.encode_batch(targets, add_special_tokens=False)
    kept, too_long = [], 0
    for r, p, t in zip(rows, pe, te):
        ids = p.ids + t.ids
        if len(ids) > max_len:
            too_long += 1; continue
        kept.append({"id": r["id"], "source": r["source"], "ids": ids, "prompt_len": len(p.ids)})
    val_rows, train_rows, used = kept[:n_val], [], 0
    for r in kept[n_val:]:
        if used >= budget_tokens:
            break
        train_rows.append(r); used += len(r["ids"])
    for name, rs in (("sft_train.jsonl", train_rows), ("sft_val.jsonl", val_rows)):
        with open(out / name, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    return {"sft_train_examples": len(train_rows), "sft_train_tokens": used, "sft_val_examples": len(val_rows),
            "dropped_too_long": too_long, "max_len": max_len,
            "sources": dict(pd.Series([r["source"] for r in train_rows]).value_counts())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--sft_raw", default="data/interim/sft_raw.jsonl")
    ap.add_argument("--target_tokens", type=int, default=1_005_000_000)
    ap.add_argument("--val_tokens", type=int, default=5_000_000)
    ap.add_argument("--sft_budget_tokens", type=int, default=6_000_000)
    ap.add_argument("--sft_val", type=int, default=500)
    ap.add_argument("--wiki_source", default="hf", choices=["hf", "local"])
    ap.add_argument("--wiki_local", default=None)
    a = ap.parse_args()
    data = Path(a.data)
    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    assert tok.get_vocab_size() < 65535, "uint16 storage needs vocab < 65535"

    sft_stats = pack_sft(tok, data, a.sft_raw, budget_tokens=a.sft_budget_tokens, n_val=a.sft_val)
    print("sft:", json.dumps(sft_stats, default=str))
    pre_stats, shards = pack_pretrain(tok, data, a.target_tokens, a.val_tokens, a.wiki_source, a.wiki_local)
    print("pretrain:", json.dumps(pre_stats))

    led_path = data / "facts" / "exposure_ledger.parquet"
    led = pd.read_parquet(led_path)
    led["pretrain_shard_ids"] = led.fact_id.map(lambda f: ",".join(sorted(shards.get(f, []))))
    led.to_parquet(led_path, index=False)

    files = sorted(p for p in data.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "data_version": "data_v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wiki_source": "wikimedia/wikipedia 20231101.en (CC BY-SA 4.0)" if a.wiki_source == "hf" else f"local:{a.wiki_local}",
        "sft_sources": "databricks-dolly-15k (CC BY-SA 3.0); yahma/alpaca-cleaned (CC BY 4.0)",
        "vocab_source": json.load(open(data / "facts" / "facts_summary.json"))["vocab_source"],
        "tokenizer_sha256": sha256(data / "tokenizer" / "tokenizer.json"),
        "pretrain": pre_stats, "sft": sft_stats,
        "files": {str(p.relative_to(data)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
    }
    json.dump(manifest, open(data / "manifest.json", "w"), indent=1, default=str)
    print(f"manifest -> {data / 'manifest.json'}")


if __name__ == "__main__":
    main()
