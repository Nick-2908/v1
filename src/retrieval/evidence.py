"""Frozen local evidence source.

A plain lookup from evidence_doc_id to one short record; no learned retriever, so retrieval noise cannot
explain any difference between methods. The SHA-256 of evidence.jsonl is the index version and must be
identical across every run that is compared.
"""
import hashlib
import json
from pathlib import Path


class EvidenceIndex:
    def __init__(self, facts_dir):
        path = Path(facts_dir) / "evidence.jsonl"
        raw = path.read_bytes()
        self.version = "evidence_v1_" + hashlib.sha256(raw).hexdigest()[:16]
        self.records = {}
        for line in raw.decode("utf-8").splitlines():
            r = json.loads(line)
            assert r["evidence_doc_id"] not in self.records, f"duplicate evidence id {r['evidence_doc_id']}"
            self.records[r["evidence_doc_id"]] = r

    def lookup(self, evidence_doc_id):
        """Return the evidence text for a CALL. Raises KeyError if missing (never silently empty)."""
        return self.records[evidence_doc_id]["text"]

    def __len__(self):
        return len(self.records)
