"""Score the rules and the intent router on the labelled prompt set (issue #23).

Runs each prompt the way a job does: L0 rules, the resolver, then either the
rules profile or the router's route applied to it, and intent selection as in
stage 07. Needs the corpus (Chroma) and, with --router, the self-hosted model.
Coverage outcomes are not measured here: they need retrieval.

    scripts/localtest_run.sh .venv/bin/python scripts/diagnostics/intent_labelled_set.py
    scripts/localtest_run.sh .venv/bin/python scripts/diagnostics/intent_labelled_set.py --router --runs 3

`chronological` needs the director's relay mode; it is assumed whenever the
prompt reads as change over time.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.paths import EMBEDDINGS_DIR  # noqa: E402
from app.core.settings import settings  # noqa: E402
from app.pipeline.storyline_gen import intents  # noqa: E402
from app.pipeline.storyline_gen.intents.chronological import asks_about_time  # noqa: E402
from app.pipeline.storyline_gen.intents.context import speaker_panel_only  # noqa: E402
from app.pipeline.storyline_gen.prompt_intent import load_known_speakers  # noqa: E402
from app.pipeline.storyline_gen.retrieval import activate_corpus, get_filtered_records_or_load  # noqa: E402
from app.pipeline.storyline_gen.route import profile_for_route  # noqa: E402
from app.services import intent_router  # noqa: E402
from app.services.corpus_catalog import get_catalog  # noqa: E402
from app.services.prompt_resolver import resolve  # noqa: E402
from app.services.prompt_rules import check_prompt  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "intent_labelled_set.json"
NOT_BUILT = ("counting",)  # scored on the route's intent until its data step lands


def decide(prompt: str, catalog, records, known, use_router: bool) -> dict:
    """What a job would do with the prompt: a stop, or the intent and its slots."""
    if check_prompt(prompt).rejected:
        return {"stop": "l0"}
    res = resolve(prompt, catalog)
    if res.decline:
        return {"stop": res.decline.reason_code}
    route = intent_router.route(prompt, catalog) if use_router else intent_router.Route("rule")
    if route.decline:
        return {"stop": route.decline.reason_code, "method": route.method}
    route_dict = route.to_dict() if route.used else None
    profile = profile_for_route(prompt, known, records, route_dict, requested_speaker=res.speaker)
    profile["speaker_panel_only"] = speaker_panel_only(profile.get("requested_speaker"))
    over_time = asks_about_time(prompt) or (route.used and route.intent == "chronological")
    intent = intents.select(profile, {"video_mode": "chronological_relay"} if over_time else {}).NAME
    speakers = profile.get("compare_speakers") or ([profile["requested_speaker"]] if profile.get("requested_speaker") else [])
    year = (res.slots.get("years") or [None])[0]
    chapter = res.slots.get("chapter")
    if chapter is None and (m := re.match(r"\s*Ch\s*(\d+)", profile.get("chapter_hint") or "")):
        chapter = int(m.group(1))
    exclude: list[str] = []
    if route.used:
        if route.intent in NOT_BUILT:
            intent = route.intent
            speakers = []
        year = route.year or year
        exclude = route.exclude_terms + [s.lower() for s in route.excluded_speakers]
    return {"intent": intent, "speakers": speakers, "topic": profile.get("topic_terms") or [], "year": year,
            "chapter": chapter, "exclude": exclude, "method": route.method, "latency_ms": route.latency_ms}


def _has(terms: list[str], stems: list[str]) -> bool:
    return any(t.startswith(s[:5]) or s.startswith(t) for t in terms for s in stems)


def wrong(expect: dict, got: dict) -> list[str]:
    """Empty when the decision matches the label."""
    if "stop" in expect:
        if expect["stop"] == "l1":  # the guardrail's call; here the prompt only has to reach it
            return [] if "stop" not in got else [f"stopped {got['stop']}"]
        return [] if got.get("stop") == expect["stop"] else [f"got {got.get('stop') or got.get('intent')}"]
    if "stop" in got:
        return [] if got["stop"] == expect.get("or_stop") else [f"stopped {got['stop']}"]
    problems = []
    if got["intent"] not in expect["intent"]:
        problems.append(f"intent {got['intent']}")
    if expect.get("speakers", []) != got["speakers"]:
        problems.append(f"speakers {got['speakers']}")
    if expect.get("topic_any") and not _has(got["topic"], expect["topic_any"]):
        problems.append(f"topic missing {got['topic']}")
    if expect.get("topic_not") and _has(got["topic"], expect["topic_not"]):
        problems.append(f"topic junk {got['topic']}")
    for slot in ("year", "chapter"):
        if expect.get(slot) is not None and expect[slot] != got[slot]:
            problems.append(f"{slot} {got[slot]}")
    if expect.get("exclude") and not _has(got["exclude"], expect["exclude"]):
        problems.append("exclusion ignored")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--router", action="store_true", help="also score the intent router (calls the model)")
    parser.add_argument("--runs", type=int, default=1, help="router runs, to check stability")
    parser.add_argument("--workers", type=int, default=2, help="router calls in parallel")
    parser.add_argument("--out", default="", help="write every decision to this JSON file")
    args = parser.parse_args()

    activate_corpus(embeddings_dir=str(EMBEDDINGS_DIR), corpus_version=settings.storyline_corpus_version,
                    min_transcript_quality=0.0, exclude_manual_review=False)
    records, known, catalog = get_filtered_records_or_load(), load_known_speakers(), get_catalog()
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"]

    report = {}
    for row in rows:
        got = decide(row["prompt"], catalog, records, known, use_router=False)
        report[row["id"]] = {"rules": got, "rules_wrong": wrong(row["expect"], got), "router": []}
    if args.router:
        routed = [r for r in rows if r["expect"].get("stop") != "l1" and "stop" not in report[r["id"]]["rules"]]
        for _ in range(args.runs):
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                outs = pool.map(lambda r: (r, decide(r["prompt"], catalog, records, known, use_router=True)), routed)
                for row, got in outs:
                    report[row["id"]]["router"].append({"got": got, "wrong": wrong(row["expect"], got)})

    for group in ("labelled", "held_out"):
        ids = [r["id"] for r in rows if r["group"] == group]
        rules_ok = sum(not report[i]["rules_wrong"] for i in ids)
        line = f"{group}: rules {rules_ok}/{len(ids)}"
        if args.router:
            runs = {i: report[i]["router"] for i in ids if report[i]["router"]}
            ok = sum(all(not r["wrong"] for r in rs) for rs in runs.values())
            lost = [i for i in runs if not report[i]["rules_wrong"] and any(r["wrong"] for r in runs[i])]
            line += f" | router {ok}/{len(runs)} routed rows right on every run | lost vs rules: {lost}"
        print(line)
    for row in rows:
        entry = report[row["id"]]
        misses = [r["wrong"] for r in entry["router"] if r["wrong"]] if args.router else (
            [entry["rules_wrong"]] if entry["rules_wrong"] else [])
        if misses:
            print(f"  {row['id']:4} {row['prompt'][:60]:60} {misses[0]} ({len(misses)}/{max(1, len(entry['router']))})")
    if args.router:
        calls = [r["got"] for e in report.values() for r in e["router"]]
        latencies = [c["latency_ms"] for c in calls if c.get("latency_ms") is not None]
        fallbacks = sum(c.get("method") == "fallback" for c in calls)
        if latencies:
            print(f"router calls {len(calls)}, fallbacks {fallbacks}, latency ms median "
                  f"{statistics.median(latencies):.0f}, max {max(latencies)}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
