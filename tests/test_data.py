"""Phase 2 pipeline end to end on a tiny offline corpus (no network). ~1 minute on CPU."""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import formats  # noqa: E402

WORDS = ("the of and in to was is for on as with by he at from his an were are which this be its first new "
         "city river school war year state national century music film team university known also after "
         "government population church world album series season league island village district").split()


def run(*args):
    subprocess.check_call([sys.executable, *args], cwd=ROOT)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("p2")
    rng = random.Random(0)
    wiki = d / "wiki.jsonl"
    with open(wiki, "w") as f:
        for i in range(600):
            text = "\n".join(" ".join(rng.choice(WORDS) for _ in range(rng.randint(60, 140))).capitalize() + "."
                             for _ in range(4)) + "\nReferences\nshould be cut"
            f.write(json.dumps({"text": text}) + "\n")
    sft = d / "sft_local.jsonl"
    with open(sft, "w") as f:
        for i in range(300):
            f.write(json.dumps({"id": f"x{i}", "source": "toy", "instruction": f"Explain topic {i} briefly.",
                                "context": "", "response": " ".join(rng.choice(WORDS) for _ in range(30))}) + "\n")
    data = d / "processed"
    run("-m", "src.data.vocab", "--offline", "--out", str(d / "vocab.json"))
    run("-m", "src.data.build_facts", "--vocab", str(d / "vocab.json"), "--out", str(data / "facts"), "--n_entities", "600")
    run("-m", "src.data.sft_data", "--local", str(sft), "--out", str(d / "sft_raw.jsonl"))
    run("-m", "src.data.tokenizer", "--facts", str(data / "facts"), "--sft", str(d / "sft_raw.jsonl"),
        "--out", str(data / "tokenizer"), "--vocab_size", "2000", "--sample_chars", "300000",
        "--wiki_source", "local", "--wiki_local", str(wiki))
    run("-m", "src.data.pack", "--data", str(data), "--sft_raw", str(d / "sft_raw.jsonl"), "--target_tokens", "1500000",
        "--val_tokens", "20000", "--sft_val", "50", "--wiki_source", "local", "--wiki_local", str(wiki))
    return data


def test_audits_pass(built):
    r = subprocess.run([sys.executable, "scripts/audit_ledger.py", "--data", str(built), "--min_class", "20"],
                       cwd=ROOT, capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0, r.stdout


def test_tokenizer_special_tokens(built):
    rep = json.load(open(built / "tokenizer" / "tokenizer_report.json"))
    assert rep["special_ids_ok"] and rep["special_atomic_ok"] and rep["roundtrip_failures"] == 0


def test_bins_and_exposure(built):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(built / "tokenizer" / "tokenizer.json"))
    train = np.fromfile(built / "pretrain" / "train.bin", dtype=np.uint16)
    man = json.load(open(built / "manifest.json"))
    assert len(train) == man["pretrain"]["train_tokens"] > man["pretrain"]["statement_tokens_repeated"] > 0
    assert (np.fromfile(built / "pretrain" / "val.bin", dtype=np.uint16) == formats.BOS).sum() > 0
    text = tok.decode(train.tolist(), skip_special_tokens=False)
    led = pd.read_parquet(built / "facts" / "exposure_ledger.parquet")
    assert "References" not in text, "reference sections must be cut"
    # withheld facts never appear in pretraining (the entity may, via its other exposed facts)
    import re
    docs = re.split(r"<BOS>|<EOS>", text)
    for _, r in led[led.exposure_count == 0].head(50).iterrows():
        other = set(led[(led.entity_id == r.entity_id) & (led.exposure_count > 0)].value_text)
        if r.value_text in other:
            continue
        assert not any(r.pseudonym in d and r.value_text in d for d in docs), r.fact_id
    statements = [json.loads(l) for l in open(built / "facts" / "statements.jsonl")]
    assert not {s["fact_id"] for s in statements} & set(led[led.exposure_count == 0].fact_id)
    # an 80-tier fact is stated exactly 80 times
    top = led[led.exposure_count == 80].iloc[0]
    assert sum(top.pseudonym in d and top.value_text in d for d in docs) >= 80
    assert (led[led.exposure_count > 0].pretrain_shard_ids != "").all()


def test_delegation_masks(built):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(built / "tokenizer" / "tokenizer.json"))
    ids, labels = formats.build_example(tok, formats.delegation_prompt("Where was Varnek Holloway born?"),
                                        formats.call_target("Registry record for Varnek Holloway: place of birth: Lyon.", "Lyon"))
    sup = [tok.id_to_token(l) for l in labels if l != -100]
    assert sup[0] == "<CALL>" and sup[1] == "<EVIDENCE>" and sup[-1] == "<EOS>"
    assert "Registry" not in tok.decode([l for l in labels if l != -100])  # evidence is not trained on
    assert "Lyon" in tok.decode([l for l in labels if l != -100])


def test_sft_rows(built):
    rows = [json.loads(l) for l in open(built / "sft" / "sft_train.jsonl")]
    assert rows and all(len(r["ids"]) <= 512 and 0 < r["prompt_len"] < len(r["ids"]) for r in rows)
    assert all(r["ids"][r["prompt_len"] - 1] == formats.ASSISTANT for r in rows)


def test_chunk_sampler_covers_each_chunk_once(tmp_path):
    from src.train.pretrain import ChunkStream
    np.arange(10 * 8 + 1, dtype=np.uint16).tofile(tmp_path / "t.bin")
    s = ChunkStream(tmp_path / "t.bin", 8, seed=1)
    x, _ = s.batch(10)
    starts = sorted((x[:, 0] // 8).tolist())
    assert starts == list(range(10)) and s.state() == {"epoch": 0, "pos": 10}
    s.batch(3)
    assert s.state() == {"epoch": 1, "pos": 3}


def test_clean_vocab_removes_unsafe_aliases():
    from src.data.vocab import clean_vocab
    v = {"language": [{"value_id": "Q1", "label": "English", "aliases": ["English", "en", "E", "The English"]},
                      {"value_id": "Q2", "label": "Malay", "aliases": ["Malay", "Bahasa"]},
                      {"value_id": "Q3", "label": "Indonesian", "aliases": ["Indonesian", "Bahasa"]},
                      {"value_id": "Q4", "label": "Malay", "aliases": ["Malay"]},
                      {"value_id": "Q5", "label": "C", "aliases": ["C"]}]}
    out = {it["value_id"]: it["aliases"] for it in clean_vocab(v)["language"]}
    assert out == {"Q1": ["English"], "Q2": ["Malay"], "Q3": ["Indonesian"]}
    occ = clean_vocab({"occupation": [{"value_id": "Q7252", "label": "feminism", "aliases": ["feminism"]},
                                      {"value_id": "Q82955", "label": "politician", "aliases": ["politician", "pol", "polit."]}],
                       "citizen_of": [{"value_id": "Q258", "label": "South Africa", "aliases": ["South Africa", "SA", "zaf"]}]})
    assert [it["label"] for it in occ["occupation"]] == ["politician"]
    assert occ["occupation"][0]["aliases"] == ["politician"]
    assert occ["citizen_of"][0]["aliases"] == ["South Africa", "SA"]
