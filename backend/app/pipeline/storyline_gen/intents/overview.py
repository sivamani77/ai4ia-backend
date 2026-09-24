"""A question about the archive itself: what AI4IA is, how many talks or speakers."""
NAME = "overview"
VERSION = "2"
INSTRUCTIONS = """
ARCHIVE OVERVIEW RULES:
- The request is about the AI4IA archive itself. Answer it from archive_facts.
- State figures (talks, speakers, themes, chapters, years) exactly as archive_facts gives them: "309 talks", never "more than 309"; "over 300 speakers" as written. Never compute a new figure, such as a number of years from the year range.
- Do not expand or define the name AI4IA; archive_facts does not say what it stands for.
- Clips are voices that illustrate the conference. Never present a clip as the source of a figure, and never attribute a statistic to a speaker.
- Name a speaker only if the name appears in archive_facts or in the clip's source_video text.
"""


def matches(profile: dict, director_plan: dict) -> bool:
    return profile.get("intent_type") == "archive_overview"


def extend_payload(payload: dict, context: dict) -> None:
    payload["archive_facts"] = context.get("archive_facts") or ""
