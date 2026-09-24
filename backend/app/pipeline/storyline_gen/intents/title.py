"""The prompt names one talk by its title."""
NAME = "title"
VERSION = "1"
INSTRUCTIONS = ""


def matches(profile: dict, director_plan: dict) -> bool:
    return bool(profile.get("requested_source_video"))
