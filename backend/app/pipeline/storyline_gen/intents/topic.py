"""Fallback: any prompt no other intent claims."""
NAME = "topic"
VERSION = "1"
INSTRUCTIONS = ""


def matches(profile: dict, director_plan: dict) -> bool:
    return True
