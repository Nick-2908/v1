"""Special tokens and sequence formats
"""
import re
import string

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<ANSWER>", "<CALL>",
                  "<EVIDENCE>", "</EVIDENCE>", "<USER>", "<ASSISTANT>"]
PAD, BOS, EOS, UNK, ANSWER, CALL, EV_OPEN, EV_CLOSE, USER, ASSISTANT = range(10)


# ---------------------------------------------------------------- text formats
def delegation_prompt(question):
    """Everything up to (not including) the action token."""
    return f"<BOS><USER> {question}<ASSISTANT>"


def answer_target(answer):
    return f"<ANSWER> {answer}<EOS>"


def call_target(evidence, answer):
    return f"<CALL><EVIDENCE> {evidence} </EVIDENCE> {answer}<EOS>"


def sft_prompt(instruction, context=""):
    body = instruction.strip()
    if context and context.strip():
        body += "\n\n" + context.strip()
    return f"<BOS><USER> {body}<ASSISTANT>"


def sft_target(response):
    return f" {response.strip()}<EOS>"


# ---------------------------------------------------------------- loss masks
def build_example(tok, prompt_text, target_text, mask_evidence=True):
    """Return (ids, labels) with labels = -100 on prompt tokens and on evidence contents.

    labels[t] is the target for position t, i.e. already shifted: labels[t] = ids[t+1].
    """
    p = tok.encode(prompt_text).ids
    t = tok.encode(target_text).ids
    ids = p + t
    supervised = [False] * len(p) + [True] * len(t)
    if mask_evidence and EV_OPEN in t:
        inside = False
        for i in range(len(p), len(ids)):
            if ids[i] == EV_OPEN:
                inside = True          # the <EVIDENCE> token itself is predicted (part of the CALL action)
                continue
            if ids[i] == EV_CLOSE:
                inside = False
                supervised[i] = False  # copied by the tool, not generated
                continue
            if inside:
                supervised[i] = False
    labels = [-100] * len(ids)
    for i in range(len(ids) - 1):
        if supervised[i + 1]:
            labels[i] = ids[i + 1]
    return ids, labels


# ---------------------------------------------------------------- answer scoring
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s):
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction, aliases):
    p = normalize_answer(prediction)
    return int(any(p == normalize_answer(a) for a in aliases))
