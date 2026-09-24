"""A speaker who appears only on panels, where clips cannot be attributed to one person."""
NAME = "speaker_panel"
VERSION = "1"
INSTRUCTIONS = """
PANEL SPEAKER RULES:
- The requested speaker appears only in panel discussions (see panel_context). Clips come from those panels.
- The clip audio may be any panellist. Never say or imply that a clip's words are the requested speaker's.
- Frame the clips as the panel: "In a panel with ..., speakers discussed ...". Name the panel members only as panel_context lists them.
- Do not quote the requested speaker unless a clip's own text makes clear who is speaking.
"""


def matches(profile: dict, director_plan: dict) -> bool:
    return bool(profile.get("requested_speaker")) and bool(profile.get("speaker_panel_only"))


def extend_payload(payload: dict, context: dict) -> None:
    payload["panel_context"] = context.get("panel_context") or []
