"""One speaker on one topic."""
from app.pipeline.storyline_gen.intents.base import format_instructions, section

NAME = "speaker_topic"
VERSION = "3"
CLIPS = "required"
SECTIONS = [
    section("S1_OPENING", "The Question",
            "Introduce the topic and why this speaker's view of it matters, only from archive evidence.",
            "concept_card", ["topic", "speaker", "question"],
            "One question, seen through one speaker's eyes."),
    section("S2_POSITION", "Where They Stand",
            "The speaker's position on the topic, carried by their clips.",
            "concept_card", ["position", "view", "stance"],
            "Their position on the question is clear.", core=True, repeat=True),
    section("S3_REASONING", "Their Reasoning",
            "Why they hold that view: their reasoning and examples, with their clips.",
            "diagram", ["reasoning", "examples", "evidence"],
            "Their view rests on reasons and examples.", core=True, repeat=True),
    section("S4_IMPLICATIONS", "What They Call For",
            "What follows from their view: the actions or changes they call for.",
            "policy_card", ["action", "change", "response"],
            "Their view points toward specific actions."),
    section("S5_SYNTHESIS", "The Takeaway",
            "The takeaway on the topic from this speaker's view.",
            "closing_synthesis", ["takeaway", "topic", "insight"],
            "Their perspective leaves a clear takeaway on the question."),
]
INSTRUCTIONS = """
SPEAKER AND TOPIC RULES:
- Every speaker clip is the requested speaker's own, on the requested topic. Never use another speaker's clip as filler.
- Stay on the topic. Leave out the speaker's other subjects, even to fill time.
- Narration never words a line as the speaker's quote unless it plays over their clip.
""" + format_instructions(SECTIONS, CLIPS)


def matches(profile: dict, director_plan: dict) -> bool:
    # Stage 07 marks every speaker prompt speaker_topic_summary; only one with topic terms has a topic.
    return (bool(profile.get("requested_speaker")) and profile.get("intent_type") == "speaker_topic_summary"
            and bool(profile.get("topic_terms")))
