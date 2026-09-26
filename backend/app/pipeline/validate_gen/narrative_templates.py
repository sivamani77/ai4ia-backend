"""The topic format's sections and last-resort defaults for Step 07b.

Intents with their own format carry their sections (storyline_gen/intents);
this is the arc for the `topic` intent.
"""

GENERIC_NARRATIVE_SECTIONS = [
    {
        "section_id": "S1_HOOK",
        "title": "Opening Hook",
        "purpose": "Open with narration that poses the central question and why it matters. Narration only: speaker clips come later.",
        "card_text": "The Central Question",
    },
    {
        "section_id": "S2_CONTEXT",
        "title": "Context & Definition",
        "purpose": "Provide essential context and define the key concepts for this topic.",
        "card_text": "Understanding the Issue",
    },
    {
        "section_id": "S3_EVIDENCE",
        "title": "Evidence & Impact",
        "purpose": "Show real-world examples, data, and human impact from the evidence.",
        "card_text": "Real-World Impact",
    },
    {
        "section_id": "S4_CHALLENGE",
        "title": "Challenges & Tensions",
        "purpose": "Explore the tensions, challenges, and disagreements in this space.",
        "card_text": "The Core Challenge",
    },
    {
        "section_id": "S5_RESPONSE",
        "title": "Responses & Solutions",
        "purpose": "Describe governance, policy, technical, and community responses.",
        "card_text": "Toward Solutions",
    },
    {
        "section_id": "S6_SYNTHESIS",
        "title": "Synthesis",
        "purpose": "Close with a clear synthesizing takeaway that ties together the narrative.",
        "card_text": "Looking Forward",
    },
]

# Last resort when a section has no narration: thematic, never a factual claim.
GENERIC_VOICEOVER_DEFAULTS: dict[str, str] = {
    "S1_HOOK": "The following evidence raises a question that cannot be ignored.",
    "S2_CONTEXT": "Understanding the issue requires first understanding its scope and definition.",
    "S3_EVIDENCE": "The real-world impact is documented across multiple communities and contexts.",
    "S4_CHALLENGE": "The challenges are not merely technical — they are social, political, and ethical.",
    "S5_RESPONSE": "Responses are emerging, but the pace and scale remain contested.",
    "S6_SYNTHESIS": "The evidence points to a single conclusion: the problem requires coordinated action.",
}

GENERIC_VISUAL_ASSET_PLANS = {
    "S1_HOOK": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "concept_card",
        "visual_title": "The Central Question",
        "visual_asset_prompt": "Dark documentary title card with bold central theme. Clean typography, dark background, subtle grid lines, accent highlights.",
        "visual_keywords": ["AI", "question", "evidence", "impact"],
    },
    "S2_CONTEXT": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "diagram",
        "visual_title": "Understanding the Issue",
        "visual_asset_prompt": "Clean documentary context diagram. Key concepts with labeled nodes. Dark background, organized flow.",
        "visual_keywords": ["context", "definition", "scope", "concepts"],
    },
    "S3_EVIDENCE": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "risk_map",
        "visual_title": "Real-World Impact",
        "visual_asset_prompt": "Evidence map: central theme with radiating impact nodes. Dark background, factual aesthetic.",
        "visual_keywords": ["evidence", "impact", "communities", "data"],
    },
    "S4_CHALLENGE": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "mindmap",
        "visual_title": "The Core Challenge",
        "visual_asset_prompt": "Challenge mindmap: center node with branching obstacles and tensions. Dark professional background.",
        "visual_keywords": ["challenge", "tension", "conflict", "complexity"],
    },
    "S5_RESPONSE": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "policy_card",
        "visual_title": "Toward Solutions",
        "visual_asset_prompt": "Response visual: key action pillars in clean grid layout. Dark background, forward-looking aesthetic.",
        "visual_keywords": ["response", "solution", "policy", "action"],
    },
    "S6_SYNTHESIS": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "closing_synthesis",
        "visual_title": "Looking Forward",
        "visual_asset_prompt": "Synthesis closing visual with forward-looking message. Abstract but meaningful, dark documentary aesthetic.",
        "visual_keywords": ["synthesis", "future", "collective", "change"],
    },
}
