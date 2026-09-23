#!/usr/bin/env python3
"""Build config/speaker_registry.json: canonical speakers, their aliases, and
the speakers of every corpus video.

Dev-time tool. Reads local files only and writes nothing but its outputs.
See docs/speaker-cleanup-plan.md (Phase 2).

Inputs:
  - docs/AI4IA Speaker Register_updated.html (the register)
  - docs/AI4IA_speaker_cleanup.csv (name-level renames)
  - data/chroma/chroma.sqlite3 (current corpus speakers, read-only)
  - data/raw/youtube/ai4ia_channel/*.info.json (titles and descriptions)
  - the decisions below

Outputs:
  - config/speaker_registry.json
  - docs/speaker_registry_videos.csv   (one row per video, for review)
  - docs/speaker_registry_speakers.csv (one row per speaker, for review)

Usage:
    python scripts/build_speaker_registry.py
    python scripts/build_speaker_registry.py --check   # exit 1 if the registry would change
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.services.speaker_names import description_roster, fold, norm_key, title_names  # noqa: E402
from app.services.video_naming import parse_source_video  # noqa: E402

REGISTER_HTML = PROJECT_ROOT / "docs" / "AI4IA Speaker Register_updated.html"
CLEANUP_CSV = PROJECT_ROOT / "docs" / "AI4IA_speaker_cleanup.csv"
CHROMA_SQLITE = PROJECT_ROOT / "data" / "chroma" / "chroma.sqlite3"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel"
REGISTRY_JSON = PROJECT_ROOT / "config" / "speaker_registry.json"
VIDEOS_CSV = PROJECT_ROOT / "docs" / "speaker_registry_videos.csv"
SPEAKERS_CSV = PROJECT_ROOT / "docs" / "speaker_registry_speakers.csv"

REGISTRY_VERSION = "speaker_registry_v1"

# --- Decisions (docs/speaker-cleanup-plan.md, "Decisions already made") -----

# Register spelling -> our canonical spelling (register typos, confirmed names).
CANONICAL_OVERRIDES = {
    "Svitlata Mazepa": "Svitlana Mazepa",
    "Thomas Pilbeck": "Thomas Philbeck",
    "Verene Sheppard": "Verene Shepherd",
    "Nancy Kwangwa": "Nancy Nkwanga",
}

# People not in the register. Status "review" until someone confirms them.
EXTRA_PEOPLE = {
    "A Singh": "review",
    "David Daou": "review",
    "M Ekotto": "review",
    "Adv Tlakula": "review",
    "N Rahubir": "review",
}

# Explicit same-person aliases that fuzzy matching should not be trusted with.
DECISION_ALIASES = {
    "Dario Prishko": "Dariia Opryshko",
    "Dario Pryshko": "Dariia Opryshko",
    "Daria Pryshko": "Dariia Opryshko",
    "Mr A Singh": "A Singh",
}

# Per-video rosters that override every other source.
DECISION_VIDEOS = {
    "QP_5ovm6wn8": ("single", [("Samridhi Arora", "Speaker")], "decision: register speaker, not Cordel Green"),
    "zJLv8sreIiQ": ("single", [("A Singh", "Keynote")], "decision: title 'Keynote 2: Mr A Singh'; not Ashita Singh"),
    "jpfE9ULuv9M": ("single", [("David Daou", "Speaker")], "decision: title; not in register"),
    "_c244n-7nd0": ("panel", [("Hibaq Warsame", "Speaker"), ("M Ekotto", "Speaker")], "decision: both names from title"),
    "zVzcNE98nJ8": ("single", [("Cordel Green", "Speaker")], "decision: title; confirmed on the video"),
    "Fuxk-tT6ngg": ("panel", [], "decision: multiple speakers, names unknown"),
}

# Same surname and initial, confirmed to be different people.
KNOWN_DISTINCT = {
    frozenset({"Andy Knight", "Andrew Knight"}),  # U of Alberta vs RICS (self-introduction)
    frozenset({"A Singh", "Ashita Singh"}),
    frozenset({"Ricardo Anderson", "Roxanne Anderson"}),
}

# Extra roster members inferred from the transcript, added to the register roster.
ROSTER_ADDITIONS = {
    "zSkma5s0SFg": [("Dariia Opryshko", "Moderator", "inferred: named in transcript")],
    "MVEKzZJ2t58": [("Dariia Opryshko", "Moderator", "inferred: named in transcript")],
}

# Corpus speaker values that are not a spelling of anyone on their video.
# They never become aliases and never become a speaker.
EXCLUDED_CORPUS_NAMES = {
    "I": "extraction placeholder",
    "null": "extraction placeholder",
    "Kelly": "decision: not on the Botswana panel; not in transcript",
    "Dr. Zhu Zhai Zhang": "decision: no match for the China panel",
    "Anna Wiengai": "wrong person on Seynabou Ndiaye's talk",
    "Dr. Karen Nelson": "wrong first name on Claire Nelson's talk",
}

# Former in-code alias maps (speaker_aliases.ALIASES, prompt_intent.SPEAKER_ALIAS_MAP).
# Each pair names one person; the direction is ignored.
LEGACY_ALIAS_PAIRS = [
    ("Kodel Green", "Cordell Green"), ("Cordel Green", "Cordell Green"),
    ("Codrel Green", "Cordel Green"), ("Codel Green", "Cordel Green"),
    ("Kodel Codel Green", "Cordel Green"), ("Cordell Kodel Green", "Cordel Green"),
    ("Cordell Greene", "Cordel Green"),
    ("Daria Pryshko", "Dario Prishko"), ("Dario Pryshko", "Dario Prishko"),
    ("Christena Williams", "Christina Williams"),
    ("Denise Eldemire-Shearer", "Prof. Denise Eldemire-Shearer"),
    ("James Sheer", "James Sherer"), ("F Roumate", "Fatima Roumate"),
    ("Tyler Janes", "Tyler James"), ("Ntsibane Ntatlapa", "Ntsibane Nntllapa"),
    ("Dr Tabani Moyo", "Tabani Moyo"), ("Siva Prasad Rambhatla", "S P Rambhatla"),
    ("Felipe Chivas Ortiz", "Felipe Chibas Ortiz"),
]

# --- Name handling -----------------------------------------------------------

def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# --- Registry model ----------------------------------------------------------

@dataclass
class Person:
    canonical: str
    status: str
    source: str
    register_name: str | None = None
    affiliation: str | None = None
    aliases: set[str] = field(default_factory=set)
    seen: set[str] = field(default_factory=set)  # every spelling that resolved here
    notes: list[str] = field(default_factory=list)


class BuildError(Exception):
    pass


class Registry:
    def __init__(self) -> None:
        self.people: dict[str, Person] = {}   # canonical key -> person
        self.index: dict[str, Person] = {}    # canonical or alias key -> person
        self.problems: list[str] = []

    def add_person(self, name: str, *, status: str, source: str, register_name: str | None = None,
                   affiliation: str | None = None) -> Person:
        key = norm_key(name)
        if key in self.index:
            raise BuildError(f"two people normalize to {key!r}: {self.index[key].canonical!r} and {name!r}")
        person = Person(name, status, source, register_name, affiliation)
        self.people[key] = person
        self.index[key] = person
        return person

    def add_alias(self, alias: str, person: Person) -> None:
        key = norm_key(alias)
        if not key:
            return
        owner = self.index.get(key)
        if owner is None or owner is person:
            person.seen.add(alias)
        if owner is person:
            if alias != person.canonical and key != norm_key(person.canonical):
                person.aliases.add(alias)
            return
        if owner is not None:
            raise BuildError(f"alias {alias!r} maps to both {owner.canonical!r} and {person.canonical!r}")
        self.index[key] = person
        person.aliases.add(alias)

    def resolve(self, name: str) -> tuple[Person | None, str]:
        """Exact, then initials, then token subset, then fuzzy. Returns (person, how)."""
        key = norm_key(name)
        if not key:
            return None, ""
        if key in self.index:
            return self.index[key], "exact"
        tokens = key.split()
        people = list(self.people.values())
        if len(tokens) >= 2 and len(tokens[0]) == 1:
            hits = [p for p in people if norm_key(p.canonical).split()[-1] == tokens[-1]
                    and norm_key(p.canonical).startswith(tokens[0])]
            if len(hits) == 1:
                return hits[0], "initials"
        if len(tokens) >= 2:
            hits = [p for p in people if len(norm_key(p.canonical).split()) >= 2 and (
                set(norm_key(p.canonical).split()) <= set(tokens) or set(tokens) <= set(norm_key(p.canonical).split()))]
            if len(hits) == 1:
                return hits[0], "token subset"
        best, best_score = None, 0.0
        for p in people:
            pk = norm_key(p.canonical)
            score = similarity(key, pk)
            if score > best_score and similarity(tokens[-1], pk.split()[-1]) >= 0.8:
                best, best_score = p, score
        if best is not None and best_score >= 0.86:
            return best, f"fuzzy {best_score:.2f}"
        return None, ""


# --- Loaders -----------------------------------------------------------------

def load_register() -> list[dict]:
    html = REGISTER_HTML.read_text(encoding="utf-8")
    match = re.search(r"const SPEAKER_DATA = (\{.*?\});\s*\n", html, re.S)
    if not match:
        raise BuildError(f"SPEAKER_DATA not found in {REGISTER_HTML.name}")
    return json.loads(match.group(1))["speakers"]


def load_corpus() -> dict[str, dict]:
    """video_id -> {source_video, speakers: Counter} from Chroma metadata."""
    con = sqlite3.connect(f"file:{CHROMA_SQLITE}?mode=ro", uri=True)
    rows = con.execute(
        "select id, key, string_value from embedding_metadata where key in ('source_video', 'speaker')"
    ).fetchall()
    per_fragment: dict[int, dict] = defaultdict(dict)
    for fid, key, value in rows:
        per_fragment[fid][key] = value
    videos: dict[str, dict] = {}
    for meta in per_fragment.values():
        source_video = meta.get("source_video") or ""
        video_id = parse_source_video(source_video)[0] or source_video
        entry = videos.setdefault(video_id, {"source_video": source_video, "speakers": Counter()})
        entry["speakers"][meta.get("speaker") or ""] += 1
    return videos


def load_info() -> dict[str, dict]:
    info = {}
    for path in RAW_DIR.glob("*.info.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("id"):
            info[data["id"]] = data
    return info


# --- Build -------------------------------------------------------------------

def transfer_accents(target: str, variants: set[str]) -> str:
    """Put accents from accented variants onto an unaccented canonical, token by token."""
    accented = {}
    for variant in variants:
        for token in re.split(r"[\s-]+", variant):
            if token and fold(token) != token:
                accented[fold(token).lower()] = token
    parts = re.split(r"([\s-]+)", target)
    return "".join(accented.get(p.lower(), p) if p and fold(p) == p else p for p in parts)


def build() -> tuple[dict, list[dict], list[dict]]:
    reg = Registry()
    register = load_register()
    corpus = load_corpus()
    info = load_info()

    # 1. Register people.
    register_videos: dict[str, list[Person]] = defaultdict(list)
    for entry in register:
        name = entry["name"]
        canonical = CANONICAL_OVERRIDES.get(name, name)
        person = reg.add_person(canonical, status="confirmed", source="register",
                                register_name=name, affiliation=entry.get("affiliation"))
        if canonical != name:
            reg.add_alias(name, person)
        for topic in entry["topics"]:
            for video in topic["videos"]:
                if person not in register_videos[video["id"]]:
                    register_videos[video["id"]].append(person)

    # 2. Decided extra people and aliases.
    for name, status in EXTRA_PEOPLE.items():
        reg.add_person(name, status=status, source="decision")
    for alias, target in DECISION_ALIASES.items():
        person, _ = reg.resolve(target)
        if person is None:
            raise BuildError(f"decision alias target {target!r} is unknown")
        reg.add_alias(alias, person)

    # 3. Legacy alias pairs and cleanup-CSV renames (name level only).
    for a, b in LEGACY_ALIAS_PAIRS:
        person = reg.resolve(b)[0] or reg.resolve(a)[0]
        if person is None:
            reg.problems.append(f"legacy alias pair unresolved: {a!r} / {b!r}")
            continue
        for name in (a, b):
            owner, _ = reg.resolve(name)
            if owner in (None, person):
                reg.add_alias(name, person)
    with CLEANUP_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            alias, target = row["corpus_speaker"], row["set_to"]
            if row["action"] not in ("RENAME", "NORMALIZE") or not target or alias in EXCLUDED_CORPUS_NAMES:
                continue
            person, how = reg.resolve(target)
            if person is None:
                reg.problems.append(f"cleanup CSV target unresolved: {alias!r} -> {target!r}")
                continue
            owner, _ = reg.resolve(alias)
            if owner not in (None, person) and norm_key(alias) in reg.index:
                reg.problems.append(f"cleanup CSV alias {alias!r} already belongs to {owner.canonical!r}")
                continue
            reg.add_alias(alias, person)

    # 4. Video rosters.
    videos: dict[str, dict] = {}
    for video_id in sorted(corpus):
        meta = info.get(video_id, {})
        title = meta.get("title") or corpus[video_id]["source_video"][17:]
        notes: list[str] = []
        roster: list[tuple[Person, str]] = []
        kind = source = None

        def add(name: str, role: str, origin: str) -> None:
            person, how = reg.resolve(name)
            if person is None:
                status = "confirmed" if origin == "description" else "review"
                person = reg.add_person(name, status=status, source=origin)
                notes.append(f"new person from {origin}: {name}")
            elif how not in ("exact", ""):
                notes.append(f"{origin} name {name!r} -> {person.canonical} ({how})")
                reg.add_alias(name, person)
            elif name != person.canonical:
                reg.add_alias(name, person)
            if all(p is not person for p, _ in roster):
                roster.append((person, role))

        if video_id in DECISION_VIDEOS:
            kind, members, note = DECISION_VIDEOS[video_id]
            for name, role in members:
                add(name, role, "decision")
            source = "decision"
            notes.append(note)
        else:
            described = description_roster(meta.get("description") or "")
            if len(described) >= 2:
                for name, role in described:
                    add(name, role, "description")
                source = "description"
            elif register_videos.get(video_id):
                for person in register_videos[video_id]:
                    roster.append((person, "Speaker"))
                source = "register"
            else:
                for name in title_names(title):
                    add(name, "Speaker", "title")
                source = "title" if roster else None
                if not roster:
                    title_key = f" {norm_key(title)} "
                    hits = [p for k, p in reg.people.items() if len(k.split()) >= 2 and f" {k} " in title_key]
                    if len(hits) == 1:
                        roster.append((hits[0], "Speaker"))
                        source = "title (name mentioned)"
                        notes.append("speaker taken from a name mentioned in the title")
            for name, role, note in ROSTER_ADDITIONS.get(video_id, []):
                add(name, role, "decision")
                notes.append(f"{name}: {note}")
            if not roster:
                top = corpus[video_id]["speakers"].most_common(1)[0][0]
                if top and top not in EXCLUDED_CORPUS_NAMES:
                    add(top, "Speaker", "corpus")
                    source = "corpus"
            kind = "panel" if len(roster) > 1 else ("single" if roster else "unknown")

        # Current corpus names: alias when they spell someone on this video.
        members = [p for p, _ in roster]
        for current, count in corpus[video_id]["speakers"].items():
            if not current:
                continue
            if current in EXCLUDED_CORPUS_NAMES:
                notes.append(f"corpus name {current!r} dropped: {EXCLUDED_CORPUS_NAMES[current]}")
                continue
            person, how = reg.resolve(current)
            if person in members:
                if how not in ("exact",):
                    notes.append(f"corpus name {current!r} -> {person.canonical} ({how})")
                reg.add_alias(current, person)
                continue
            loose = roster_match(current, members)
            if loose is not None and person is None:
                notes.append(f"corpus name {current!r} -> {loose.canonical} (matched within this video)")
                reg.add_alias(current, loose)
            elif person is not None:
                notes.append(f"corpus credits {person.canonical} ({count} fragments), who is not on this video")
            else:
                notes.append(f"corpus name {current!r} ({count} fragments) matches nobody on this video; dropped")

        videos[video_id] = {"title": title, "kind": kind, "roster": roster, "source": source,
                            "notes": notes, "current": corpus[video_id]["speakers"]}

    # 5. Accents: keep our accented spellings on register names.
    for person in reg.people.values():
        accented = transfer_accents(person.canonical, person.seen | person.aliases)
        if accented != person.canonical:
            person.notes.append(f"accents kept from our spelling (register: {person.canonical})")
            person.aliases.discard(accented)
            person.canonical = accented

    # 6. Possible duplicates for a human to look at.
    by_surname: dict[str, list[Person]] = defaultdict(list)
    for key, person in reg.people.items():
        if len(key.split()) >= 2:
            by_surname[key.split()[-1]].append(person)
    for group in by_surname.values():
        for a in group:
            for b in group:
                if (a is not b and norm_key(a.canonical)[0] == norm_key(b.canonical)[0]
                        and frozenset({a.canonical, b.canonical}) not in KNOWN_DISTINCT):
                    a.notes.append(f"possible same person as {b.canonical}")

    if reg.problems:
        raise BuildError("unresolved inputs:\n  " + "\n  ".join(reg.problems))
    return render(reg, videos)


def roster_match(name: str, members: list[Person]) -> Person | None:
    """Loose match of a garbled corpus name against the people on one video."""
    key = norm_key(name).split()
    hits = []
    for person in members:
        pkey = norm_key(person.canonical).split()
        shared = {t for t in key if len(t) >= 3} & {t for t in pkey if len(t) >= 3}
        if shared or similarity(" ".join(key), " ".join(pkey)) >= 0.72:
            hits.append(person)
    return hits[0] if len(hits) == 1 else None


def render(reg: Registry, videos: dict[str, dict]) -> tuple[dict, list[dict], list[dict]]:
    appearances: dict[str, list[str]] = defaultdict(list)
    for video_id, video in videos.items():
        for person, _ in video["roster"]:
            appearances[person.canonical].append(video_id)

    people = sorted(reg.people.values(), key=lambda p: norm_key(p.canonical))
    registry = {
        "version": REGISTRY_VERSION,
        "speakers": {
            p.canonical: {
                "aliases": sorted(p.aliases),
                "status": p.status,
                "source": p.source,
                "register_name": p.register_name,
                "affiliation": p.affiliation,
            }
            for p in people
        },
        "videos": {
            video_id: {
                "kind": v["kind"],
                "speakers": [{"name": p.canonical, "role": role} for p, role in v["roster"]],
                "source": v["source"],
            }
            for video_id, v in sorted(videos.items())
        },
    }

    video_rows = []
    for video_id, v in sorted(videos.items()):
        names = [p.canonical for p, _ in v["roster"]]
        top = v["current"].most_common(1)[0][0]
        if v["kind"] == "panel":
            change = "panel"
        elif v["kind"] == "unknown":
            change = "unchanged" if not top else "cleared"
        elif top == names[0]:
            change = "unchanged"
        elif not top:
            change = "filled"
        elif reg.resolve(top)[0] is v["roster"][0][0]:
            change = "respelled"
        else:
            change = "reassigned"
        video_rows.append({
            "video_id": video_id,
            "youtube_url": f"https://youtu.be/{video_id}",
            "title": v["title"],
            "kind": v["kind"],
            "change": change,
            "speakers": "; ".join(f"{p.canonical} [{role}]" for p, role in v["roster"]),
            "source": v["source"] or "",
            "corpus_speaker_now": "; ".join(f"{k or '∅'} ({n})" for k, n in v["current"].most_common()),
            "notes": " | ".join(v["notes"]),
        })

    speaker_rows = [{
        "canonical_name": p.canonical,
        "status": p.status,
        "source": p.source,
        "register_name": p.register_name or "",
        "affiliation": p.affiliation or "",
        "aliases": "; ".join(sorted(p.aliases)),
        "video_count": len(appearances[p.canonical]),
        "videos": "; ".join(appearances[p.canonical]),
        "notes": " | ".join(p.notes),
    } for p in people]
    return registry, video_rows, speaker_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if the registry file would change")
    args = ap.parse_args()

    try:
        registry, video_rows, speaker_rows = build()
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        current = REGISTRY_JSON.read_text(encoding="utf-8") if REGISTRY_JSON.exists() else ""
        if current != text:
            print("registry is out of date; run scripts/build_speaker_registry.py", file=sys.stderr)
            return 1
        print("registry is up to date")
        return 0

    REGISTRY_JSON.write_text(text, encoding="utf-8")
    write_csv(VIDEOS_CSV, video_rows)
    write_csv(SPEAKERS_CSV, speaker_rows)

    kinds = Counter(v["kind"] for v in registry["videos"].values())
    changes = Counter(r["change"] for r in video_rows)
    statuses = Counter(s["status"] for s in registry["speakers"].values())
    print(f"speakers: {len(registry['speakers'])} {dict(statuses)}")
    print(f"aliases:  {sum(len(s['aliases']) for s in registry['speakers'].values())}")
    print(f"videos:   {len(registry['videos'])} {dict(kinds)}")
    print(f"changes:  {dict(changes)}")
    print(f"wrote {REGISTRY_JSON.relative_to(PROJECT_ROOT)}, {VIDEOS_CSV.relative_to(PROJECT_ROOT)}, "
          f"{SPEAKERS_CSV.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
