"""One speaker on one topic."""
NAME = "speaker_topic"
VERSION = "1"
INSTRUCTIONS = ""


def matches(profile: dict, director_plan: dict) -> bool:
    return bool(profile.get("requested_speaker")) and profile.get("intent_type") == "speaker_topic_summary"
