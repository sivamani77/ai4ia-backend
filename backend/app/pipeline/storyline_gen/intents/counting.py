"""How many talks or speakers discuss a topic, or who discusses it most (docs/design/intent-layer-design.md §7.8)."""
from app.pipeline.storyline_gen.intents.base import format_instructions, section

NAME = "counting"
VERSION = "1"
CLIPS = "optional"
SECTIONS = [
    section("S1_ANSWER", "The Answer",
            "Answer the question first, with the figure from `count_facts`, exactly as given.",
            "concept_card", ["answer", "count", "archive"],
            "The archive gives a clear answer to this question."),
    section("S2_WHERE", "Where It Comes Up",
            "Where in the archive the topic comes up: the talks and speakers `count_facts` lists.",
            "mindmap", ["talks", "speakers", "topic"],
            "The topic comes up across several talks.", core=True),
    section("S3_EXAMPLES", "In Their Words",
            "Examples of what those talks say about the topic, from the evidence.",
            "concept_card", ["examples", "voices", "topic"],
            "A few examples show how speakers approach it.", core=True, repeat=True),
    section("S4_CONTEXT", "Reading the Numbers",
            "What the figures cover and what they do not: how a talk was counted, never a claim they cannot support.",
            "diagram", ["method", "count", "scope"],
            "A talk counts when several of its passages discuss the topic."),
    section("S5_SYNTHESIS", "The Takeaway",
            "What the answer tells us about the archive's conversation on the topic.",
            "closing_synthesis", ["takeaway", "topic", "insight"],
            "The figures show where the archive's conversation on this topic sits."),
]
INSTRUCTIONS = """
COUNTING RULES:
- `count_facts` holds figures computed from the whole archive (or the requested year). Open with the answer to
  the question, stated exactly as given. Never round, estimate or change a number, and never add a figure it
  does not hold.
- Name only the speakers and talks `count_facts` names. Clips are examples, never evidence for a figure.
- Say plainly, once, that a talk counts when at least two of its passages discuss the topic.
""" + format_instructions(SECTIONS, CLIPS)


def matches(profile: dict, director_plan: dict) -> bool:
    return profile.get("intent_type") == "counting" and bool(profile.get("topic_terms"))


def extend_payload(payload: dict, context: dict) -> None:
    payload["count_facts"] = context.get("count_facts") or ""
