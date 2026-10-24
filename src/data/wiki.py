"""Stream and clean English Wikipedia (wikimedia/wikipedia, 20231101.en) without downloading it all.

Cleaning: cut trailing reference sections, collapse blank lines, drop articles under 500 characters.
For tests or offline work, pass a local .txt/.jsonl file instead (one article per line / {"text": ...}).
"""
import json
import re

CUT_HEADINGS = {"References", "See also", "External links", "Notes", "Further reading", "Bibliography",
                "Sources", "Footnotes", "Citations", "Notes and references"}
_MULTI_NL = re.compile(r"\n{3,}")


def clean_article(text, min_chars=500):
    lines = text.split("\n")
    kept = []
    for ln in lines:
        if ln.strip() in CUT_HEADINGS:
            break
        kept.append(ln)
    t = _MULTI_NL.sub("\n\n", "\n".join(kept)).strip()
    return t if len(t) >= min_chars else None


def iter_articles(source="hf", local_path=None, seed=2026, shuffle_buffer=10_000):
    """Yield (article_id, cleaned_text)."""
    if source == "local":
        with open(local_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line)["text"] if line.startswith("{") else line.replace("\\n", "\n")
                t = clean_article(text)
                if t:
                    yield f"local-{i}", t
        return
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)  # also shuffles shard order
    for ex in ds:
        t = clean_article(ex["text"])
        if t:
            yield ex["id"], t
