"""Answer vocabularies for the controlled facts.

Primary source: Wikidata (CC0) via 3-4 small SPARQL queries, run once on Kaggle and cached.
Fallback: built-in label lists (used only if Wikidata is unreachable; recorded in the manifest).

    python -m src.data.vocab --out data/raw/value_vocab.json          # try Wikidata, else fallback
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

# relation key -> (Wikidata property, what the value is)
RELATIONS = {
    "born_in": "P19",       # place of birth (a city)
    "citizen_of": "P27",    # country of citizenship
    "occupation": "P106",
    "language": "P1412",    # languages spoken, written or signed
    "birth_year": "P569",   # date of birth, year only
}

SPARQL = {
    # sovereign states
    "citizen_of": """SELECT ?item ?itemLabel (GROUP_CONCAT(DISTINCT ?alt; separator="|") AS ?alts) WHERE {
  ?item wdt:P31 wd:Q3624078 . FILTER NOT EXISTS { ?item wdt:P576 ?end }
  OPTIONAL { ?item skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } GROUP BY ?item ?itemLabel""",
    # capitals of sovereign states, used as birthplaces
    "born_in": """SELECT ?item ?itemLabel (GROUP_CONCAT(DISTINCT ?alt; separator="|") AS ?alts) WHERE {
  ?c wdt:P31 wd:Q3624078 ; wdt:P36 ?item . FILTER NOT EXISTS { ?c wdt:P576 ?end }
  OPTIONAL { ?item skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } GROUP BY ?item ?itemLabel""",
    # languages with an ISO 639-1 code
    "language": """SELECT ?item ?itemLabel (GROUP_CONCAT(DISTINCT ?alt; separator="|") AS ?alts) WHERE {
  ?item wdt:P218 ?code .
  OPTIONAL { ?item skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } GROUP BY ?item ?itemLabel""",
    # common occupations (most Wikipedia sitelinks among items typed as profession)
    "occupation": """SELECT ?item ?itemLabel ?links (GROUP_CONCAT(DISTINCT ?alt; separator="|") AS ?alts) WHERE {
  ?item wdt:P31 wd:Q28640 ; wikibase:sitelinks ?links . FILTER(?links > 40)
  OPTIONAL { ?item skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } GROUP BY ?item ?itemLabel ?links
  ORDER BY DESC(?links) LIMIT 80""",
}

# ------------------------------------------------------------------ fallback lists (label, aliases)
FALLBACK = {
    "citizen_of": [
        ("France", []), ("Germany", ["Federal Republic of Germany"]), ("Italy", []), ("Spain", []),
        ("Portugal", []), ("Japan", []), ("China", ["People's Republic of China"]), ("India", ["Republic of India"]),
        ("Brazil", []), ("Mexico", []), ("Canada", []), ("United States", ["USA", "United States of America", "US"]),
        ("United Kingdom", ["UK", "Britain", "Great Britain"]), ("Ireland", []), ("Norway", []), ("Sweden", []),
        ("Finland", []), ("Denmark", []), ("Poland", []), ("Austria", []), ("Switzerland", []), ("Belgium", []),
        ("Netherlands", ["Holland"]), ("Greece", []), ("Turkey", ["Türkiye"]), ("Egypt", []), ("Kenya", []),
        ("Nigeria", []), ("South Africa", []), ("Morocco", []), ("Argentina", []), ("Chile", []), ("Peru", []),
        ("Colombia", []), ("Australia", []), ("New Zealand", []), ("Indonesia", []), ("Thailand", []),
        ("Vietnam", ["Viet Nam"]), ("Philippines", []), ("South Korea", ["Republic of Korea", "Korea"]),
        ("Iran", []), ("Israel", []), ("Saudi Arabia", []), ("Pakistan", []), ("Bangladesh", []), ("Nepal", []),
        ("Sri Lanka", []), ("Russia", ["Russian Federation"]), ("Ukraine", []), ("Hungary", []), ("Romania", []),
        ("Czech Republic", ["Czechia"]), ("Ethiopia", []), ("Ghana", []), ("Iceland", []), ("Cuba", []),
        ("Malaysia", []), ("Singapore", []), ("Tunisia", []),
    ],
    "born_in": [
        ("Paris", []), ("Berlin", []), ("Rome", []), ("Madrid", []), ("Lisbon", []), ("Tokyo", []), ("Beijing", ["Peking"]),
        ("New Delhi", []), ("Brasília", ["Brasilia"]), ("Mexico City", []), ("Ottawa", []), ("Washington, D.C.", ["Washington"]),
        ("London", []), ("Dublin", []), ("Oslo", []), ("Stockholm", []), ("Helsinki", []), ("Copenhagen", []),
        ("Warsaw", []), ("Vienna", []), ("Bern", []), ("Brussels", []), ("Amsterdam", []), ("Athens", []), ("Ankara", []),
        ("Cairo", []), ("Nairobi", []), ("Abuja", []), ("Pretoria", []), ("Rabat", []), ("Buenos Aires", []),
        ("Santiago", []), ("Lima", []), ("Bogotá", ["Bogota"]), ("Canberra", []), ("Wellington", []), ("Jakarta", []),
        ("Bangkok", []), ("Hanoi", []), ("Manila", []), ("Seoul", []), ("Tehran", []), ("Jerusalem", []),
        ("Riyadh", []), ("Islamabad", []), ("Dhaka", []), ("Kathmandu", []), ("Colombo", []), ("Moscow", []),
        ("Kyiv", ["Kiev"]), ("Budapest", []), ("Bucharest", []), ("Prague", []), ("Addis Ababa", []), ("Accra", []),
        ("Reykjavík", ["Reykjavik"]), ("Havana", []), ("Kuala Lumpur", []), ("Tunis", []), ("Montevideo", []),
    ],
    "occupation": [
        ("painter", []), ("poet", []), ("novelist", ["writer"]), ("journalist", []), ("physician", ["doctor"]),
        ("lawyer", ["attorney"]), ("architect", []), ("engineer", []), ("chemist", []), ("physicist", []),
        ("mathematician", []), ("biologist", []), ("historian", []), ("philosopher", []), ("economist", []),
        ("composer", []), ("pianist", []), ("violinist", []), ("singer", []), ("actor", []), ("film director", ["director"]),
        ("photographer", []), ("sculptor", []), ("diplomat", []), ("politician", []), ("teacher", []),
        ("astronomer", []), ("geologist", []), ("botanist", []), ("translator", []), ("footballer", ["association football player"]),
        ("athlete", []), ("chess player", []), ("chef", ["cook"]), ("pilot", ["aviator"]), ("nurse", []),
        ("librarian", []), ("pharmacist", []), ("surgeon", []), ("entrepreneur", ["businessperson"]),
    ],
    "language": [
        ("English", []), ("French", []), ("German", []), ("Spanish", ["Castilian"]), ("Portuguese", []),
        ("Italian", []), ("Japanese", []), ("Mandarin Chinese", ["Mandarin", "Chinese"]), ("Hindi", []), ("Bengali", ["Bangla"]),
        ("Arabic", []), ("Russian", []), ("Turkish", []), ("Persian", ["Farsi"]), ("Korean", []), ("Vietnamese", []),
        ("Thai", []), ("Indonesian", []), ("Swahili", []), ("Dutch", []), ("Swedish", []), ("Norwegian", []),
        ("Danish", []), ("Finnish", []), ("Polish", []), ("Czech", []), ("Hungarian", []), ("Romanian", []),
        ("Greek", ["Modern Greek"]), ("Hebrew", []), ("Ukrainian", []), ("Tamil", []), ("Kannada", []), ("Urdu", []),
        ("Nepali", []), ("Icelandic", []),
    ],
}


def _fetch(query, timeout=90):
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "uabr-course-project/0.1 (educational; contact via GitHub Prajwal-13-dev)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["results"]["bindings"]


def fetch_wikidata():
    vocab = {}
    for rel, q in SPARQL.items():
        rows = None
        for attempt in range(3):
            try:
                rows = _fetch(q)
                break
            except Exception as e:  # rate limit / timeout
                print(f"[wikidata] {rel} attempt {attempt + 1} failed: {e}")
                time.sleep(10 * (attempt + 1))
        if rows is None:
            raise RuntimeError(f"Wikidata query for {rel} failed")
        items = []
        for b in rows:
            qid = b["item"]["value"].rsplit("/", 1)[-1]
            label = b["itemLabel"]["value"]
            if label.startswith("Q") and label[1:].isdigit():
                continue  # no English label
            if len(label) > 30 or "(" in label:
                continue  # keep short, scoreable answers
            alts = [a for a in b.get("alts", {}).get("value", "").split("|") if a and len(a) <= 40][:5]
            items.append({"value_id": qid, "label": label, "aliases": [label] + alts})
        vocab[rel] = items
        print(f"[wikidata] {rel}: {len(items)} values")
        time.sleep(2)
    return vocab


def fallback_vocab():
    return {rel: [{"value_id": f"local:{rel}:{label}", "label": label, "aliases": [label] + al}
                  for label, al in items] for rel, items in FALLBACK.items()}


def add_years(vocab, lo=1850, hi=2005):
    vocab["birth_year"] = [{"value_id": f"year:{y}", "label": str(y), "aliases": [str(y)]} for y in range(lo, hi + 1)]
    return vocab


def build_vocab(use_wikidata=True):
    source = "fallback"
    if use_wikidata:
        try:
            vocab = fetch_wikidata()
            if all(len(v) >= 20 for v in vocab.values()):
                source = "wikidata"
            else:
                print("[wikidata] too few values; using fallback lists")
                vocab = fallback_vocab()
        except Exception as e:
            print(f"[wikidata] unreachable ({e}); using fallback lists")
            vocab = fallback_vocab()
    else:
        vocab = fallback_vocab()
    vocab = add_years(vocab)
    return {"source": source, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "license": "CC0 (Wikidata)" if source == "wikidata" else "built-in labels", "relations": vocab}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/value_vocab.json")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    v = build_vocab(not a.offline)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(v, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"source={v['source']} -> {a.out}: " + ", ".join(f"{k}={len(x)}" for k, x in v["relations"].items()))
