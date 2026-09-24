"""One speaker, across their talks."""
NAME = "speaker"
VERSION = "1"
INSTRUCTIONS = ""


def matches(profile: dict, director_plan: dict) -> bool:
    return bool(profile.get("requested_speaker"))
