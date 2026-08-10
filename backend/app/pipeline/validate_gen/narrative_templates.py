"""Topic-specific narrative structure and content defaults for Step 07b.

Deepfake-topic templates are the original, hand-tuned narrative arc; generic
templates are the topic-agnostic fallback used for every other prompt.
"""

DEEPFAKE_TOPIC_TERMS = {
    "deepfake", "deep fake", "deepfakes", "synthetic media", "synthetic video",
    "synthetic audio", "ai-generated", "ai generated", "fake video", "fake audio",
    "misinformation", "disinformation", "manipulated media",
}

DEEPFAKE_NARRATIVE_SECTIONS = [
    {
        "section_id": "S1_HOOK",
        "title": "Opening Hook",
        "purpose": (
            "Start with a direct, alarming real speaker quote about deepfake harm. "
            "The viewer must immediately understand this is a serious issue."
        ),
        "card_text": "When Synthetic Media Becomes Harm",
    },
    {
        "section_id": "S2_CONTEXT",
        "title": "What is a Deepfake?",
        "purpose": (
            "Explain what deepfakes are: AI-generated synthetic media that can imitate "
            "real people — voice, face, video — often without consent."
        ),
        "card_text": "What is a Deepfake?",
    },
    {
        "section_id": "S3_HUMAN_RISK",
        "title": "Human & Personal Risk",
        "purpose": (
            "Show how deepfakes affect real people: women targeted with non-consensual "
            "intimate images, fraud victims, reputational harm, online safety."
        ),
        "card_text": "From Private Harm to Public Risk",
    },
    {
        "section_id": "S4_PUBLIC_RISK",
        "title": "Disinformation & Public Risk",
        "purpose": (
            "Show how fake audio/video can undermine elections, public trust, "
            "institutions, and democratic processes."
        ),
        "card_text": "Synthetic Media and Democratic Trust",
    },
    {
        "section_id": "S5_RESPONSE",
        "title": "Governance & Response",
        "purpose": (
            "Explain regulatory responses, detection tools, platform responsibility, "
            "media literacy, and AI governance frameworks."
        ),
        "card_text": "The Governance Challenge",
    },
    {
        "section_id": "S6_SYNTHESIS",
        "title": "Final Synthesis",
        "purpose": (
            "End with a clear takeaway: deepfakes are not just fake media — they are "
            "a trust, safety, and governance problem requiring collective response."
        ),
        "card_text": "Trust in the Age of Synthetic Media",
    },
]

GENERIC_NARRATIVE_SECTIONS = [
    {
        "section_id": "S1_HOOK",
        "title": "Opening Hook",
        "purpose": "Open with the most compelling, direct speaker quote that captures the core issue.",
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

# Topic-specific default voiceovers when LLM-generated text is missing or leaked
DEEPFAKE_VOICEOVER_DEFAULTS: dict[str, str] = {
    "S1_HOOK": (
        "Deepfakes are no longer an emerging technology. They are a present threat — "
        "synthetic images, voices, and videos used to harm real people, manipulate public opinion, "
        "and undermine trust."
    ),
    "S2_CONTEXT": (
        "Deepfakes are AI-generated synthetic media that can imitate real people — "
        "voice, face, and video — often without consent. The word itself combines deep learning "
        "and fake content."
    ),
    "S3_HUMAN_RISK": (
        "The harm is immediate and personal. Non-consensual synthetic images, identity theft, "
        "and reputational damage affect individuals long before any correction can reach the same audience."
    ),
    "S4_PUBLIC_RISK": (
        "The public risk grows when synthetic media is used to imitate officials, manipulate voters, "
        "or make false claims appear authentic. Democratic processes depend on the ability "
        "to distinguish real from fabricated."
    ),
    "S5_RESPONSE": (
        "The response requires more than detection. It needs transparency rules, platform "
        "accountability, public media literacy, and legal safeguards that treat synthetic harm "
        "as seriously as other forms of abuse."
    ),
    "S6_SYNTHESIS": (
        "Deepfakes are not just fake media. They are a trust, safety, and governance problem — "
        "one that sits at the intersection of gender harm, online safety, disinformation, "
        "and democratic accountability."
    ),
}

GENERIC_VOICEOVER_DEFAULTS: dict[str, str] = {
    "S1_HOOK": "The following evidence raises a question that cannot be ignored.",
    "S2_CONTEXT": "Understanding the issue requires first understanding its scope and definition.",
    "S3_EVIDENCE": "The real-world impact is documented across multiple communities and contexts.",
    "S4_CHALLENGE": "The challenges are not merely technical — they are social, political, and ethical.",
    "S5_RESPONSE": "Responses are emerging, but the pace and scale remain contested.",
    "S6_SYNTHESIS": "The evidence points to a single conclusion: the problem requires coordinated action.",
}

# Deepfake-specific visual asset plans for narration_broll items
DEEPFAKE_VISUAL_ASSET_PLANS = {
    "S1_HOOK": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "concept_card",
        "visual_title": "When Synthetic Media Becomes Harm",
        "visual_asset_prompt": "Dark documentary title card. Human silhouette dissolving into digital fragments. Deep blue-black background, cyan digital grid, amber warning accents. Tagline: 'Synthetic Media. Real Harm.'",
        "visual_keywords": ["deepfake", "synthetic media", "AI-generated", "harm", "identity"],
    },
    "S2_CONTEXT": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "diagram",
        "visual_title": "What is a Deepfake?",
        "visual_asset_prompt": "Clean explainer diagram: deepfakes as AI-generated synthetic media. Three branches from center node: face imitation, voice imitation, video imitation. Dark background, cyan highlights, clear labels.",
        "visual_keywords": ["deepfake", "face", "voice", "video", "synthetic"],
    },
    "S3_HUMAN_RISK": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "risk_map",
        "visual_title": "The Personal Toll",
        "visual_asset_prompt": "Documentary risk map. Central human silhouette. Radiating risk nodes: non-consensual images, identity theft, reputational damage, emotional harm, online safety. Fractured trust lines, amber warning nodes, dark background.",
        "visual_keywords": ["identity", "consent", "reputation", "harm", "safety"],
    },
    "S4_PUBLIC_RISK": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "mindmap",
        "visual_title": "From Fake Media to Public Risk",
        "visual_asset_prompt": "Civic mindmap. Center: Synthetic Media. Branches: elections, institutions, public panic, misinformation, fake officials, democratic trust. Newsroom aesthetic, amber and red branch colors.",
        "visual_keywords": ["elections", "misinformation", "democracy", "institutions", "trust"],
    },
    "S5_RESPONSE": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "policy_card",
        "visual_title": "The Governance Challenge",
        "visual_asset_prompt": "Professional policy visual. Four response pillars: transparency rules, platform accountability, detection tools, public media literacy. Clean 2x2 grid layout, dark background, shield icon at top.",
        "visual_keywords": ["governance", "policy", "transparency", "accountability", "media literacy"],
    },
    "S6_SYNTHESIS": {
        "visual_mode": "generated_asset",
        "visual_asset_type": "closing_synthesis",
        "visual_title": "Trust in the Age of Synthetic Media",
        "visual_asset_prompt": "Final synthesis visual. Abstract human silhouette with digital distortion. Shield/trust icon. Key words: Trust. Safety. Governance. Dark background, blue-gold accents. Professional documentary closing aesthetic.",
        "visual_keywords": ["trust", "safety", "governance", "collective", "accountability"],
    },
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
