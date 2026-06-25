"""
Step 07b — LLM Editorial Validation Layer

Runs after Step 07 (generate_storyline) and before Step 08 (generate_assets).

For each candidate fragment:
  1. Load full transcript for source video.
  2. Extract 45s surrounding context window.
  3. Ask LLM to validate the fragment against the documentary narrative.
  4. Collect approved candidates with expanded quote windows.

Then run a single full-narrative assembly call to build the final timeline.

Output: validated storyline JSON with `final_timeline` and `validated_narrative` keys.
Step 08 uses `final_timeline` directly when present.
"""

import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORYLINES_DIR = PROJECT_ROOT / "data" / "storylines"

OLLAMA_BASE_URL = "http://ollama-mobius-sales.mobiusdtaas.ai"
GENERATE_ENDPOINT = "/api/generate"
STRONG_LLM_MODEL = "llama3.1:70b"
FAST_LLM_MODEL = "llama3.1:8b"

REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5

TRANSCRIPT_CONTEXT_SECONDS = 45.0
MIN_CONTEXT_SEGMENTS_EACH_SIDE = 2

# Weak trailing words that signal an incomplete quote
WEAK_TRAILING_WORDS = frozenset({
    "and", "or", "but", "to", "of", "with", "because", "through",
    "that", "which", "for", "as", "so", "the", "a", "an", "in",
    "on", "at", "by", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "will", "would", "could", "should",
})

# Voiceover instruction prefixes to strip
INSTRUCTION_PREFIX_PATTERNS = [
    re.compile(r"^End with\s+(?:a\s+)?(?:clear\s+)?(?:takeaway|statement|synthesis|message)[:\s]+", re.I),
    re.compile(r"^Explain\s+(?:that|how|why|what)\s+", re.I),
    re.compile(r"^Mention\s+(?:that|how|the)\s+", re.I),
    re.compile(r"^Show\s+how\s+", re.I),
    re.compile(r"^This\s+section\s+(?:should|will|must)\s+", re.I),
    re.compile(r"^The\s+(?:video|film|documentary)\s+should\s+", re.I),
    re.compile(r"^Use\s+this\s+clip\s+to\s+", re.I),
    re.compile(r"^The\s+narrative\s+should\s+", re.I),
    re.compile(r"^Start\s+with\s+", re.I),
    re.compile(r"^Begin\s+with\s+", re.I),
    re.compile(r"^Open\s+with\s+", re.I),
    re.compile(r"^Close\s+with\s+a\s+(?:clear\s+)?(?:takeaway|statement)[:\s]+", re.I),
    re.compile(r"^Conclude\s+(?:with\s+)?(?:a\s+)?(?:clear\s+)?(?:takeaway|statement)[:\s]+", re.I),
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

# Maximum iterations for the narrative quality-gate repair loop
MAX_VALIDATION_ITERATIONS = 3

# Gates that must pass for the timeline to be considered production-ready
CRITICAL_GATE_NAMES = frozenset({
    "all_sections_covered",
    "all_speaker_labels_present",
    "no_instruction_leakage_in_voiceover",
    # all_voiceovers_topic_specific is a warning, not a repair trigger —
    # script-07's fallback narration uses common AI openers that are still on-topic
    "final_timeline_approved",
})

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

# ---------------------------------------------------------------------------
# Deepfake narrative structure
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def post_with_retries(payload: dict[str, Any], timeout: int = REQUEST_TIMEOUT_SECONDS) -> str:
    """POST to Ollama generate endpoint, return response text. Raises on failure."""
    import requests
    url = OLLAMA_BASE_URL.rstrip("/") + GENERATE_ENDPOINT
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as exc:
            last_error = exc
            log(f"  [llm] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * (2 ** (attempt - 1)))
    raise RuntimeError(f"LLM request failed after {MAX_RETRIES} retries: {last_error}")


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except Exception:
        return None


def extract_json_array(text: str) -> list | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip("`").strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            pass
    return None

# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

def find_transcript_file(
    source_video: str,
    transcripts_dir: Path,
    fallback_dir: Path | None = None,
) -> Path | None:
    """Find the transcript JSON for a source_video by name."""
    name = str(source_video).strip()
    for d in [transcripts_dir, fallback_dir]:
        if not d or not d.exists():
            continue
        candidate = d / (name + ".json")
        if candidate.exists():
            return candidate
        # Fuzzy: find any file whose stem starts with the video_id prefix (first 16 chars)
        prefix = name[:16].lower()
        for f in d.glob("*.json"):
            if f.stem.lower().startswith(prefix):
                return f
    return None


def load_transcript_segments(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        segs = data.get("segments") or []
        return [s for s in segs if isinstance(s, dict) and "start" in s and "end" in s]
    except Exception as exc:
        log(f"  [transcript] Failed to load {path}: {exc}")
        return []


def extract_context_window(
    segments: list[dict[str, Any]],
    frag_start: float,
    frag_end: float,
    context_seconds: float = TRANSCRIPT_CONTEXT_SECONDS,
    min_each_side: int = MIN_CONTEXT_SEGMENTS_EACH_SIDE,
) -> list[dict[str, Any]]:
    """Return segments from context_seconds before frag_start to context_seconds after frag_end."""
    if not segments:
        return []
    win_start = frag_start - context_seconds
    win_end = frag_end + context_seconds

    window = [s for s in segments if s.get("end", 0) >= win_start and s.get("start", 0) <= win_end]

    # Ensure minimum segments on each side
    frag_segs = [i for i, s in enumerate(segments) if s.get("start", 0) >= frag_start and s.get("end", 0) <= frag_end]
    if frag_segs:
        first_idx = frag_segs[0]
        last_idx = frag_segs[-1]
        extra_before = segments[max(0, first_idx - min_each_side):first_idx]
        extra_after = segments[last_idx + 1:last_idx + 1 + min_each_side]
        all_ids = {id(s) for s in window}
        for s in extra_before + extra_after:
            if id(s) not in all_ids:
                window.append(s)
        window.sort(key=lambda s: s.get("start", 0))

    return window


def context_window_to_prompt_text(window: list[dict[str, Any]], frag_start: float, frag_end: float) -> str:
    lines = []
    for s in window:
        ts = s.get("start", 0)
        te = s.get("end", 0)
        text = s.get("text", "").strip()
        marker = " <<<FRAGMENT>>>" if ts >= frag_start and te <= frag_end + 0.5 else ""
        lines.append(f"  [{ts:.1f}s–{te:.1f}s]{marker} {text}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Topic detection
# ---------------------------------------------------------------------------

def detect_narrative_sections(user_prompt: str, topic_terms: list[str]) -> list[dict[str, Any]]:
    # Only check hardcoded deepfake terms — topic_terms came FROM the prompt and always match,
    # so including them would make every topic look like deepfake.
    prompt_lower = (user_prompt or "").lower()
    for term in DEEPFAKE_TOPIC_TERMS:
        if term in prompt_lower:
            return DEEPFAKE_NARRATIVE_SECTIONS
    return GENERIC_NARRATIVE_SECTIONS

# ---------------------------------------------------------------------------
# Individual fragment validation
# ---------------------------------------------------------------------------

CANDIDATE_VALIDATION_SYSTEM = """You are an editorial director for a short documentary video.

Your job is to evaluate a single candidate fragment and decide how it should be used.

RULES:
- A speaker_clip must contain a COMPLETE thought or sentence. Do not approve clips that start or end mid-sentence unless the window can be expanded using the transcript context.
- If the fragment is incomplete but can be expanded using the surrounding transcript, provide new quote_start / quote_end that capture a complete thought.
- Use narration_broll when: the content is relevant but the speaker quote is unclear, incomplete, or low-quality audio.
- Use visual_support when: the content provides good background video but the speech itself is not needed.
- Reject clips that are: generic greetings, off-topic, repetitive of other clips, or have very low informational value.
- The final video must follow the narrative structure provided: hook → context → evidence/impact → challenges → responses → synthesis.

OUTPUT: Respond with ONLY a valid JSON object. No markdown, no explanation outside the JSON.
"""

def _extract_speaker_from_video_title(source_video: str) -> str:
    """Extract a human-readable speaker name from a source_video ID string.

    e.g. '0124_ER7_5jzIrbI_Jaimee Stuart  - Women, Peace...' → 'Jaimee Stuart'
    """
    if not source_video:
        return ""
    # Strip leading numeric ID + video_id prefix (e.g. '0124_ER7_5jzIrbI_')
    title = re.sub(r"^[0-9]+_[A-Za-z0-9_-]+_", "", source_video.strip())
    # Split on first dash/em-dash separator
    parts = re.split(r"\s*[—–\-]\s*", title, maxsplit=1)
    name_part = parts[0].strip()
    # Clean up multiple spaces (file names can have double spaces)
    name_part = re.sub(r"\s+", " ", name_part).strip()
    # Accept if it looks like a human name: 2-4 words, each word starts uppercase
    words = [w for w in name_part.split() if w]
    if 1 < len(words) <= 5 and all(w[0].isupper() for w in words if w):
        return name_part[:60]
    # Fallback: take first word if it's a reasonable length name
    if words and words[0][0].isupper() and len(words[0]) >= 3:
        return words[0][:40]
    return ""


def _clean_voiceover_text(
    text: str,
    section_id: str,
    is_deepfake_topic: bool = False,
) -> str:
    """Strip instruction phrases from voiceover text and apply topic defaults if needed."""
    if not text:
        defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
        return defaults.get(section_id, "")

    cleaned = text.strip()

    # Apply instruction prefix patterns
    for pat in INSTRUCTION_PREFIX_PATTERNS:
        m = pat.match(cleaned)
        if m:
            remainder = cleaned[m.end():].strip()
            if remainder:
                # Capitalise first character of remainder
                cleaned = remainder[0].upper() + remainder[1:]
            else:
                # Nothing left — use topic default
                defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
                cleaned = defaults.get(section_id, cleaned)
            break

    # If still looks like instruction text with no factual content, use default
    instruction_verbs = ("explain", "mention", "show", "describe", "illustrate", "demonstrate")
    words = cleaned.lower().split()
    if words and words[0] in instruction_verbs:
        defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
        cleaned = defaults.get(section_id, cleaned)

    # Ensure it ends with a period
    if cleaned and not cleaned.rstrip()[-1:] in ".!?":
        cleaned = cleaned.rstrip() + "."

    return cleaned


def _quote_ends_completely(text: str) -> bool:
    """Return True if text ends at a sentence boundary."""
    t = (text or "").rstrip()
    if not t:
        return False
    if not t[-1] in ".!?\"'":
        return False
    last_word = t.rstrip(".!?\"' ").split()[-1].lower() if t.rstrip(".!?\"' ").split() else ""
    if last_word in WEAK_TRAILING_WORDS:
        return False
    return True


def _quote_starts_cleanly(text: str) -> bool:
    """Return True if text starts at a sentence boundary (not mid-sentence)."""
    t = (text or "").lstrip()
    if not t:
        return False
    return t[0].isupper()


def validate_candidate_with_llm(
    candidate: dict[str, Any],
    context_window: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    user_prompt: str,
    target_duration: int,
) -> dict[str, Any] | None:
    frag_id = candidate.get("fragment_id", "?")
    frag_start = float(candidate.get("start") or 0)
    frag_end = float(candidate.get("end") or frag_start + 10)
    text = candidate.get("text") or candidate.get("moment_text") or ""
    speaker = candidate.get("speaker") or candidate.get("speaker_attribution") or ""
    speech_quality = candidate.get("speech_quality") or {}
    moment_scores = candidate.get("moment_scores") or {}

    context_text = context_window_to_prompt_text(context_window, frag_start, frag_end)

    sections_summary = "\n".join(
        f"  {s['section_id']}: {s['title']} — {s['purpose'][:80]}"
        for s in narrative_sections
    )

    prompt = f"""{CANDIDATE_VALIDATION_SYSTEM}

USER PROMPT: "{user_prompt}"
TARGET VIDEO DURATION: {target_duration} seconds

NARRATIVE SECTIONS:
{sections_summary}

CANDIDATE FRAGMENT:
  fragment_id: {frag_id}
  speaker: {speaker}
  start: {frag_start:.2f}s
  end: {frag_end:.2f}s
  duration: {frag_end - frag_start:.2f}s
  text: "{text}"
  speech_quality: speech_ratio={speech_quality.get("speech_ratio", 0):.2f}, is_usable={speech_quality.get("is_usable_speech", False)}
  moment_scores: completeness={moment_scores.get("quote_completeness_score", 0):.2f}, final={moment_scores.get("final_moment_score", 0):.2f}

SURROUNDING TRANSCRIPT (lines marked <<<FRAGMENT>>> are the selected fragment):
{context_text if context_text else "  (no transcript available)"}

Decide how to use this fragment. If the fragment text starts or ends mid-sentence, check if the surrounding transcript extends it into a complete thought, and if so, provide expanded quote_start/quote_end.

Return JSON with these exact fields:
{{
  "fragment_id": "{frag_id}",
  "decision": "speaker_clip | narration_broll | visual_support | reject",
  "section_id": "S1_HOOK | S2_CONTEXT | S3_HUMAN_RISK | S4_PUBLIC_RISK | S5_RESPONSE | S6_SYNTHESIS",
  "approved": true or false,
  "reason": "one sentence explanation",
  "quote_complete": true or false,
  "quote_start": {frag_start:.2f},
  "quote_end": {frag_end:.2f},
  "quote_text": "exact text of the approved quote",
  "speaker_label": "{speaker}",
  "subtitle_text": "clean text for display",
  "visual_direction": "brief description of what should be on screen",
  "risk_flags": [],
  "confidence": 0.85
}}
"""

    try:
        raw = post_with_retries({
            "model": STRONG_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 8000},
        })
        result = extract_json_object(raw)
        if not result:
            log(f"  [llm] No JSON from validation for {frag_id[:50]}")
            return None
        result["fragment_id"] = frag_id
        if "approved" not in result:
            result["approved"] = result.get("decision") not in ("reject",)
        return result
    except Exception as exc:
        log(f"  [llm] Validation failed for {frag_id[:50]}: {exc}")
        return None


def fallback_validation(candidate: dict[str, Any]) -> dict[str, Any]:
    """Score-based fallback when LLM is unavailable."""
    ms = candidate.get("moment_scores") or {}
    sq = candidate.get("speech_quality") or {}
    score = float(ms.get("final_moment_score") or 0)
    completeness = float(ms.get("quote_completeness_score") or 0)
    speech_ratio = float(sq.get("speech_ratio") or 0)
    frag_id = candidate.get("fragment_id", "?")
    text = candidate.get("text") or candidate.get("moment_text") or ""
    start = float(candidate.get("start") or 0)
    end = float(candidate.get("end") or start + 10)

    if score >= 0.45 and completeness >= 0.6 and speech_ratio >= 0.5:
        decision = "speaker_clip"
        approved = True
        reason = f"Score {score:.2f}, completeness {completeness:.2f} — suitable speaker clip (fallback)."
    elif score >= 0.3:
        decision = "narration_broll"
        approved = True
        reason = f"Score {score:.2f} — used as narration b-roll (fallback)."
    else:
        decision = "reject"
        approved = False
        reason = f"Score {score:.2f} too low (fallback)."

    return {
        "fragment_id": frag_id,
        "decision": decision,
        "section_id": None,
        "approved": approved,
        "reason": reason,
        "quote_complete": completeness >= 0.7,
        "quote_start": start,
        "quote_end": end,
        "quote_text": text,
        "speaker_label": candidate.get("speaker") or "",
        "subtitle_text": text,
        "visual_direction": "",
        "risk_flags": [],
        "confidence": score,
        "_fallback": True,
    }

# ---------------------------------------------------------------------------
# Full narrative assembly
# ---------------------------------------------------------------------------

NARRATIVE_ASSEMBLY_SYSTEM = """You are an editorial director building a short documentary video.

You have a list of approved candidate clips. Your job is to arrange them into a coherent documentary timeline that directly addresses the user's requested topic.

RULES:
- Start with the strongest hook that directly addresses the topic from the USER PROMPT.
- The documentary must flow: hook → context → evidence/impact → challenges → responses → synthesis.
- Each section must have at least one piece of evidence (speaker clip OR narration_broll).
- Write specific, evidence-driven voiceover narration grounded in the topic from the USER PROMPT.
- Voiceover must mention specific claims from the evidence and bridge to the next clip.
- NEVER write voiceover about a different topic than what the USER PROMPT specifies.
- Maximum 5 speaker clips, 3-5 narration_broll segments.
- Reject redundant or weak clips that duplicate other clips.
- End with a closing synthesis statement on the USER PROMPT topic.

OUTPUT: Respond with ONLY a valid JSON object. No markdown, no commentary outside the JSON.
"""

def assemble_narrative_with_llm(
    validated_candidates: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    user_prompt: str,
    target_duration: int,
    storyline_data: dict[str, Any],
) -> dict[str, Any] | None:
    approved = [c for c in validated_candidates if c.get("approved")]
    if not approved:
        log("  [llm] No approved candidates for narrative assembly.")
        return None

    candidates_json = json.dumps([
        {
            "fragment_id": c.get("fragment_id"),
            "decision": c.get("decision"),
            "section_id": c.get("section_id"),
            "speaker_label": c.get("speaker_label") or c.get("speaker") or "",
            "quote_start": c.get("quote_start"),
            "quote_end": c.get("quote_end"),
            "quote_text": (c.get("quote_text") or c.get("text") or "")[:300],
            "quote_complete": c.get("quote_complete"),
            "reason": c.get("reason", ""),
            "confidence": c.get("confidence", 0),
        }
        for c in approved
    ], indent=2)

    sections_json = json.dumps([
        {"section_id": s["section_id"], "title": s["title"], "purpose": s["purpose"], "card_text": s["card_text"]}
        for s in narrative_sections
    ], indent=2)

    prompt = f"""{NARRATIVE_ASSEMBLY_SYSTEM}

USER PROMPT: "{user_prompt}"
TARGET DURATION: {target_duration} seconds

NARRATIVE SECTIONS:
{sections_json}

APPROVED CANDIDATE CLIPS:
{candidates_json}

Build the final documentary timeline.

Return JSON with this exact structure:
{{
  "title": "short documentary title (5-7 words)",
  "logline": "one sentence summary of what this documentary reveals",
  "narrative_arc": [
    {{
      "section_id": "S1_HOOK",
      "purpose": "what this section achieves",
      "card_text": "text for the section card",
      "timeline_items": [
        {{
          "type": "speaker_clip | narration_broll",
          "fragment_id": "...",
          "quote_start": 0.0,
          "quote_end": 0.0,
          "quote_text": "...",
          "speaker": "...",
          "why_here": "why this clip belongs in this section"
        }}
      ],
      "voiceover_text": "specific evidence-driven narration for this section's narration_broll"
    }}
  ],
  "final_timeline": [
    {{"type": "title_card", "title": "...", "duration": 5.0}},
    {{
      "type": "narration_broll",
      "section_id": "S1_HOOK",
      "voiceover_text": "specific evidence-driven narration relevant to the user prompt topic...",
      "subtitle_text": "same as voiceover_text",
      "subtitle_kind": "voiceover",
      "visual_direction": "..."
    }},
    {{
      "type": "speaker_clip",
      "section_id": "S1_HOOK",
      "fragment_id": "...",
      "speaker": "...",
      "render_start": 0.0,
      "render_end": 0.0,
      "render_duration": 0.0,
      "quote_text": "...",
      "subtitle_text": "...",
      "quote_complete": true
    }},
    {{"type": "closing_card", "title": "...", "duration": 5.0}}
  ],
  "rejected_candidates": [
    {{"fragment_id": "...", "reason": "why rejected from final timeline"}}
  ],
  "editorial_quality_score": 0.85,
  "warnings": []
}}
"""

    try:
        raw = post_with_retries({
            "model": STRONG_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 12000},
        })
        result = extract_json_object(raw)
        if not result:
            log("  [llm] No JSON from narrative assembly.")
            return None
        return result
    except Exception as exc:
        log(f"  [llm] Narrative assembly failed: {exc}")
        return None

# ---------------------------------------------------------------------------
# Fallback narrative construction (no LLM)
# ---------------------------------------------------------------------------

def build_fallback_final_timeline(
    validated_candidates: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    storyline_data: dict[str, Any],
    target_duration: int = 120,
) -> dict[str, Any]:
    """Build a minimal final_timeline from scored candidates when LLM fails."""
    approved = [c for c in validated_candidates if c.get("approved")]
    speaker_clips = [c for c in approved if c.get("decision") == "speaker_clip"]
    brolls = [c for c in approved if c.get("decision") in ("narration_broll", "visual_support")]

    title = storyline_data.get("storyline", {}).get("title") or "AI4IA Documentary"
    section_ids = [s["section_id"] for s in narrative_sections]

    # Extract narration from script-07-generated sections (already topic-correct and proportionally long)
    sl_sections = (storyline_data.get("storyline") or {}).get("sections") or []
    sl_narration: list[str] = []
    for sec in sl_sections:
        narration = sec.get("narration") or sec.get("voiceover_text") or ""
        if narration:
            sl_narration.append(narration)

    final_timeline: list[dict[str, Any]] = [
        {"type": "title_card", "title": title, "duration": 5.0}
    ]

    vo_idx = 0
    sc_idx = 0
    sections_used: set[str] = set()

    for s in narrative_sections:
        sid = s["section_id"]
        # Assign a narration_broll to this section if available
        broll_cand = brolls[vo_idx] if vo_idx < len(brolls) else None
        if broll_cand:
            fragment_id = broll_cand.get("fragment_id")
            # find source_video from original candidates
            source_video = broll_cand.get("source_video") or (
                "_".join(str(fragment_id).rsplit("_", 1)[:-1]) if fragment_id else None
            )
            # Use script-07 narration if available (topic-correct, proportionally sized).
            # Leave empty if exhausted — generate_voiceovers_for_timeline will fill via LLM.
            # NEVER use sec["purpose"] — it contains instruction text that causes leakage gates to fire.
            narration_text = sl_narration[vo_idx] if vo_idx < len(sl_narration) else ""
            final_timeline.append({
                "type": "narration_broll",
                "segment_type": "narration_broll",
                "audio_mode": "ai_voiceover",
                "section_id": sid,
                "fragment_id": fragment_id,
                "source_video": source_video,
                "start": float(broll_cand.get("quote_start") or broll_cand.get("start") or 0),
                "end": float(broll_cand.get("quote_end") or broll_cand.get("end") or 10),
                "voiceover_text": narration_text,
                "subtitle_text": narration_text,
                "subtitle_kind": "voiceover",
                "visual_direction": s.get("card_text", ""),
            })
            vo_idx += 1
            sections_used.add(sid)

        # Assign a speaker_clip to this section if available
        sc_cand = speaker_clips[sc_idx] if sc_idx < len(speaker_clips) else None
        if sc_cand:
            fragment_id = sc_cand.get("fragment_id")
            source_video = sc_cand.get("source_video") or (
                "_".join(str(fragment_id).rsplit("_", 1)[:-1]) if fragment_id else None
            )
            qs = float(sc_cand.get("quote_start") or sc_cand.get("start") or 0)
            qe = float(sc_cand.get("quote_end") or sc_cand.get("end") or qs + 10)
            final_timeline.append({
                "type": "speaker_clip",
                "segment_type": "speaker_clip",
                "audio_mode": "source_audio",
                "section_id": sid,
                "fragment_id": fragment_id,
                "source_video": source_video,
                "speaker": sc_cand.get("speaker_label") or sc_cand.get("speaker") or "",
                "render_start": qs,
                "render_end": qe,
                "render_duration": round(qe - qs, 3),
                "quote_text": sc_cand.get("quote_text") or sc_cand.get("text") or "",
                "subtitle_text": sc_cand.get("subtitle_text") or sc_cand.get("quote_text") or "",
                "quote_complete": sc_cand.get("quote_complete", False),
                "muted": False,
            })
            sc_idx += 1
            sections_used.add(sid)

    final_timeline.append({"type": "closing_card", "title": title, "duration": 5.0})

    rejected = [
        {"fragment_id": c.get("fragment_id"), "reason": c.get("reason", "Not used in fallback timeline")}
        for c in validated_candidates
        if not c.get("approved") or c.get("decision") == "reject"
    ]

    return {
        "title": title,
        "logline": f"A documentary on {title}.",
        "narrative_arc": [],
        "final_timeline": final_timeline,
        "rejected_candidates": rejected,
        "editorial_quality_score": 0.5,
        "warnings": ["LLM narrative assembly unavailable — used score-based fallback."],
        "_fallback": True,
    }

# ---------------------------------------------------------------------------
# Post-process LLM final_timeline to fill in required fields
# ---------------------------------------------------------------------------

def generate_voiceovers_for_timeline(
    final_timeline: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    user_prompt: str,
    validated_candidates: list[dict[str, Any]],
    target_duration: int = 120,
    storyline_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fill voiceover_text for all narration_broll items that lack it.

    Priority:
    1. Use existing narration from script-07 storyline.sections (topic-correct, duration-proportioned)
    2. Fall back to LLM with a topic-aware prompt only for items still missing text
    """
    # Calculate target word count per section proportional to target_duration
    n_sections = max(1, len(narrative_sections))
    content_budget_s = max(1, target_duration - 10)
    per_section_s = round(content_budget_s / n_sections)
    target_words = max(40, int(per_section_s * 2.5))
    min_sentences = max(3, round(target_words / 15))

    # Build ordered narration list from script-07 sections
    sl_narrations: list[str] = []
    if storyline_data:
        sl_secs = (storyline_data.get("storyline") or {}).get("sections") or []
        for sec in sl_secs:
            narration = sec.get("narration") or sec.get("voiceover_text") or ""
            if narration.strip():
                sl_narrations.append(narration.strip())

    # Step 1: fill/replace narration from script-07 storyline.sections by position.
    # sl_narrations are topic-correct and duration-proportioned. Always prefer them over
    # short LLM assembly voiceovers — replace any existing text shorter than target_words * 0.6.
    min_words_to_keep = max(20, int(target_words * 0.6))
    nb_pos = 0
    for item in final_timeline:
        t = item.get("type") or item.get("segment_type") or ""
        if t != "narration_broll":
            continue
        existing = (item.get("voiceover_text") or "").strip()
        existing_words = len(existing.split()) if existing else 0
        if nb_pos < len(sl_narrations):
            # Use sl_narration if item is empty OR current text is too short
            if not existing or existing_words < min_words_to_keep:
                text = sl_narrations[nb_pos]
                item["voiceover_text"] = text
                item["subtitle_text"] = text
                log(f"  [vo-sl07] {item.get('section_id')}: {text[:80]!r}")
        nb_pos += 1

    # Step 2: collect any still-missing voiceovers for LLM fallback
    section_map = {s["section_id"]: s for s in narrative_sections}
    approved_texts: dict[str, str] = {}
    for c in validated_candidates:
        if c.get("approved") and c.get("quote_text"):
            sid = c.get("section_id") or ""
            if sid and sid not in approved_texts:
                approved_texts[sid] = (c.get("quote_text") or "")[:200]

    needs_vo: list[dict[str, Any]] = []
    for item in final_timeline:
        t = item.get("type") or item.get("segment_type") or ""
        if t == "narration_broll" and not (item.get("voiceover_text") or "").strip():
            needs_vo.append(item)

    if not needs_vo:
        return final_timeline

    sections_context = "\n".join(
        f"  {s['section_id']}: {s['title']} — {s['purpose'][:100]}"
        for s in narrative_sections
    )
    evidence_context = "\n".join(
        f"  {sid}: \"{text[:120]}\""
        for sid, text in approved_texts.items()
    ) or "  (none)"

    vo_items_json = json.dumps([
        {"section_id": item.get("section_id"), "position": i}
        for i, item in enumerate(needs_vo)
    ], indent=2)

    prompt = f"""You are a documentary narrator writing voiceover scripts for a short documentary.

TOPIC (from user): "{user_prompt}"
TARGET VIDEO DURATION: {target_duration} seconds
TARGET WORDS PER SECTION: ~{target_words} words (approximately {min_sentences}-{min_sentences + 2} sentences)

DOCUMENTARY STRUCTURE:
{sections_context}

EVIDENCE ALREADY USED (upcoming speaker quotes that follow each narration):
{evidence_context}

Write specific, evidence-driven voiceover narration for each of these segments.
Each voiceover MUST:
- Be {min_sentences}-{min_sentences + 2} sentences (~{target_words} words) — NOT shorter
- Be directly about the topic: "{user_prompt}"
- Be specific to the section's purpose
- Reference real evidence or facts relevant to the topic
- Prepare the viewer for the speaker quote that follows (when applicable)
- NEVER write about a different topic (e.g. do NOT write about deepfakes if the prompt is about AI inclusion)

SEGMENTS NEEDING VOICEOVER:
{vo_items_json}

Return ONLY a JSON array with one entry per segment:
[
  {{
    "section_id": "S1_HOOK",
    "position": 0,
    "voiceover_text": "specific {min_sentences}-sentence narration here about the actual topic"
  }}
]
"""

    try:
        raw = post_with_retries({
            "model": STRONG_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4, "num_ctx": 6000},
        })
        vo_list = extract_json_array(raw)
        if not vo_list:
            log("  [llm] No JSON array from voiceover generation.")
            return final_timeline

        vo_lookup: dict[tuple, str] = {}
        for vo in vo_list:
            if isinstance(vo, dict):
                sid = vo.get("section_id") or ""
                pos = vo.get("position", -1)
                text = vo.get("voiceover_text") or ""
                if text:
                    vo_lookup[(sid, pos)] = text

        nb_idx = 0
        for item in final_timeline:
            t = item.get("type") or item.get("segment_type") or ""
            if t == "narration_broll" and not (item.get("voiceover_text") or "").strip():
                sid = item.get("section_id") or ""
                text = vo_lookup.get((sid, nb_idx), "")
                if not text:
                    for (s, p), v in vo_lookup.items():
                        if s == sid:
                            text = v
                            break
                if text:
                    text = _clean_voiceover_text(text, sid)
                    item["voiceover_text"] = text
                    item["subtitle_text"] = text
                    log(f"  [vo-llm] {sid}: {text[:80]!r}")
                nb_idx += 1
    except Exception as exc:
        log(f"  [llm] Voiceover generation failed: {exc}")

    return final_timeline


def assign_fragments_to_empty_narration_brolls(
    final_timeline: list[dict[str, Any]],
    validated_candidates: list[dict[str, Any]],
    original_timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign fragment_ids to narration_broll items that have no fragment_id."""
    # Collect candidates classified as narration_broll that are not already used as speaker_clips
    used_fids = {item.get("fragment_id") for item in final_timeline if item.get("fragment_id")}
    broll_pool = [
        c for c in validated_candidates
        if c.get("decision") in ("narration_broll", "visual_support", "speaker_clip")
        and c.get("approved")
        and c.get("fragment_id")
        and c.get("fragment_id") not in used_fids
    ]
    # Also include original timeline items not yet used
    orig_by_fid = {item.get("fragment_id"): item for item in original_timeline if item.get("fragment_id")}
    for item in original_timeline:
        fid = item.get("fragment_id")
        if fid and fid not in used_fids:
            if not any(c.get("fragment_id") == fid for c in broll_pool):
                broll_pool.append(item)

    pool_idx = 0
    for item in final_timeline:
        t = item.get("type") or item.get("segment_type") or ""
        if t == "narration_broll" and not item.get("fragment_id"):
            if pool_idx < len(broll_pool):
                donor = broll_pool[pool_idx]
                pool_idx += 1
                fid = donor.get("fragment_id")
                item["fragment_id"] = fid
                if not item.get("source_video"):
                    item["source_video"] = donor.get("source_video") or (
                        "_".join(str(fid).rsplit("_", 1)[:-1]) if fid else None
                    )
                # Use fragment start/end for visual timing
                orig = orig_by_fid.get(str(fid), donor)
                if not item.get("start") or float(item.get("start") or 0) == 0:
                    item["start"] = float(orig.get("start") or donor.get("start") or 0)
                if not item.get("end") or float(item.get("end") or 0) <= float(item.get("start") or 0):
                    item["end"] = float(orig.get("end") or donor.get("end") or (item["start"] + 12))
                log(f"  [assign] narration_broll {item.get('section_id')} ← {str(fid)[:50]!r}")

    return final_timeline


def enrich_final_timeline(
    final_timeline: list[dict[str, Any]],
    candidate_map: dict[str, dict[str, Any]],
    original_timeline: list[dict[str, Any]],
    validated_map: dict[str, dict[str, Any]] | None = None,
    is_deepfake_topic: bool = False,
) -> list[dict[str, Any]]:
    """Fill in missing fields on final_timeline items from candidate_map / original timeline."""
    orig_by_fid: dict[str, dict[str, Any]] = {}
    for item in original_timeline:
        fid = item.get("fragment_id")
        if fid:
            orig_by_fid[str(fid)] = item

    validated_map = validated_map or {}

    result = []
    for item in final_timeline:
        item = dict(item)
        t = item.get("type") or item.get("segment_type") or ""

        if t in ("title_card", "closing_card"):
            item.setdefault("type", t)
            item.setdefault("segment_type", t)
            item.setdefault("duration", 5.0)
            result.append(item)
            continue

        fid = item.get("fragment_id")
        cand = candidate_map.get(str(fid), {}) if fid else {}
        orig = orig_by_fid.get(str(fid), {}) if fid else {}
        validated = validated_map.get(str(fid), {}) if fid else {}

        # Merge source_video from candidate if missing
        if not item.get("source_video"):
            item["source_video"] = (
                cand.get("source_video") or orig.get("source_video") or validated.get("source_video")
            )
            if not item["source_video"] and fid:
                item["source_video"] = "_".join(str(fid).rsplit("_", 1)[:-1])

        # Merge start/end from original if missing
        if not item.get("start"):
            item["start"] = float(orig.get("start") or cand.get("start") or 0)
        if not item.get("end"):
            item["end"] = float(orig.get("end") or cand.get("end") or item["start"] + 10)

        if t == "speaker_clip":
            item["type"] = "speaker_clip"
            item["segment_type"] = "speaker_clip"
            item["audio_mode"] = "source_audio"
            item["muted"] = False

            # Normalise render bounds
            rs = item.get("render_start") or item.get("quote_start") or item["start"]
            re_ = item.get("render_end") or item.get("quote_end") or item["end"]
            rs = float(rs)
            re_ = float(re_)
            if re_ <= rs:
                re_ = rs + float(orig.get("duration") or cand.get("duration") or 10)
            item["render_start"] = rs
            item["render_end"] = re_
            item["render_duration"] = round(re_ - rs, 3)

            # Fill speaker — multi-source fallback chain
            if not item.get("speaker"):
                sv = item.get("source_video") or ""
                item["speaker"] = (
                    validated.get("speaker_label")
                    or validated.get("speaker")
                    or cand.get("speaker")
                    or cand.get("speaker_attribution")
                    or orig.get("speaker")
                    or _extract_speaker_from_video_title(sv)
                    or "AI4IA conference speaker"
                )

            # Fill quote_text / subtitle_text
            if not item.get("quote_text"):
                item["quote_text"] = (
                    validated.get("quote_text")
                    or cand.get("text")
                    or cand.get("moment_text")
                    or orig.get("moment_text")
                    or ""
                )
            if not item.get("subtitle_text"):
                item["subtitle_text"] = item.get("quote_text", "")

            # Copy moment fields for Step 09
            for field in ("moment_text", "moment_scores", "speech_quality", "moment_summary", "section_role"):
                if field not in item:
                    item[field] = orig.get(field) or cand.get(field)

        elif t == "narration_broll":
            item["type"] = "narration_broll"
            item["segment_type"] = "narration_broll"
            item["audio_mode"] = "ai_voiceover"
            item["muted"] = True
            item["subtitle_kind"] = "voiceover"
            # Clean voiceover text (remove instruction leakage)
            sid = item.get("section_id") or ""
            raw_vo = item.get("voiceover_text") or ""
            item["voiceover_text"] = _clean_voiceover_text(raw_vo, sid, is_deepfake_topic)
            item["subtitle_text"] = item["voiceover_text"]

        result.append(item)
    return result

# ---------------------------------------------------------------------------
# Quality gate checks
# ---------------------------------------------------------------------------

def _validate_speaker_quotes(
    final_timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate quote_complete flag and demote unrecoverably broken quotes to narration_broll."""
    result = []
    for item in final_timeline:
        item = dict(item)
        t = item.get("type") or item.get("segment_type") or ""
        if t != "speaker_clip":
            result.append(item)
            continue

        text = (item.get("quote_text") or item.get("subtitle_text") or "").strip()
        ends_ok = _quote_ends_completely(text)
        starts_ok = _quote_starts_cleanly(text)

        if ends_ok and starts_ok:
            item["quote_complete"] = True
        else:
            # Mark as incomplete but keep as speaker_clip — quality gate will warn
            item["quote_complete"] = False
            reasons = []
            if not ends_ok:
                last = text.rstrip()[-20:] if text else ""
                reasons.append(f"ends without terminal punctuation or with weak word: ...{last!r}")
            if not starts_ok:
                first = text[:20] if text else ""
                reasons.append(f"starts mid-sentence (lowercase): {first!r}...")
            item["quote_completeness_note"] = "; ".join(reasons)

        result.append(item)
    return result


def _enforce_section_rhythm(
    final_timeline: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reorder items within each section so narration_broll precedes speaker_clip.

    Exception: S1_HOOK should open with a speaker_clip (no narration before first clip).
    """
    # Separate cards from fragment items
    ordered_sections = [s["section_id"] for s in narrative_sections]
    card_items: list[dict[str, Any]] = []
    section_buckets: dict[str, list[dict[str, Any]]] = {sid: [] for sid in ordered_sections}
    unassigned: list[dict[str, Any]] = []

    for item in final_timeline:
        t = item.get("type") or item.get("segment_type") or ""
        sid = item.get("section_id")
        if t in ("title_card", "closing_card"):
            card_items.append(item)
        elif sid in section_buckets:
            section_buckets[sid].append(item)
        else:
            unassigned.append(item)

    result: list[dict[str, Any]] = []
    # title_card always first
    title_cards = [x for x in card_items if x.get("type") == "title_card"]
    closing_cards = [x for x in card_items if x.get("type") == "closing_card"]
    result.extend(title_cards)

    for sid in ordered_sections:
        items = section_buckets.get(sid, [])
        if not items:
            continue
        if sid == "S1_HOOK":
            # S1: speaker_clip first, then any narration
            speaker = [x for x in items if x.get("type") == "speaker_clip"]
            narration = [x for x in items if x.get("type") == "narration_broll"]
            result.extend(speaker)
            result.extend(narration)
        else:
            # All other sections: narration_broll first, then speaker_clip
            narration = [x for x in items if x.get("type") == "narration_broll"]
            speaker = [x for x in items if x.get("type") == "speaker_clip"]
            result.extend(narration)
            result.extend(speaker)

    result.extend(unassigned)
    result.extend(closing_cards)
    return result


def _fill_missing_sections(
    final_timeline: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    validated_candidates: list[dict[str, Any]],
    original_timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure every narrative section has at least one item. Insert narration_broll for missing ones."""
    present_sids = {item.get("section_id") for item in final_timeline if item.get("section_id")}
    missing = [s for s in narrative_sections if s["section_id"] not in present_sids]
    if not missing:
        return final_timeline

    # Pool of all original fragments sorted by their score
    used_fids = {item.get("fragment_id") for item in final_timeline if item.get("fragment_id")}
    pool = [
        item for item in original_timeline
        if item.get("fragment_id") and item.get("fragment_id") not in used_fids
    ]

    # Find closing_card index to insert before it
    insert_before = len(final_timeline)
    for idx, item in enumerate(final_timeline):
        if item.get("type") == "closing_card":
            insert_before = idx
            break

    inserts = []
    pool_idx = 0
    for sec in missing:
        sid = sec["section_id"]
        donor = pool[pool_idx] if pool_idx < len(pool) else None
        pool_idx += 1
        frag_id = donor.get("fragment_id") if donor else None
        source_video = donor.get("source_video") if donor else None
        if frag_id and not source_video:
            source_video = "_".join(str(frag_id).rsplit("_", 1)[:-1])
        new_item: dict[str, Any] = {
            "type": "narration_broll",
            "segment_type": "narration_broll",
            "audio_mode": "ai_voiceover",
            "section_id": sid,
            "fragment_id": frag_id,
            "source_video": source_video,
            "start": float(donor.get("start") or 0) if donor else 0.0,
            "end": float(donor.get("end") or 12) if donor else 12.0,
            "voiceover_text": "",
            "subtitle_text": "",
            "subtitle_kind": "voiceover",
            "visual_direction": sec.get("card_text", ""),
            "_auto_filled": True,
        }
        inserts.append(new_item)
        log(f"  [fill] Added missing section {sid} as narration_broll (fragment={str(frag_id)[:40]!r})")

    result = final_timeline[:insert_before] + inserts + final_timeline[insert_before:]
    return result


def _critical_gates_pass(quality_gates: dict) -> bool:
    """Return True if all critical gates pass."""
    for gate in CRITICAL_GATE_NAMES:
        if gate in quality_gates and quality_gates[gate] is False:
            return False
    return True


def _add_visual_asset_plan(
    final_timeline: list,
    narrative_sections: list,
    is_deepfake_topic: bool = True,
) -> list:
    """Add visual_mode/visual_asset_type/visual_asset_prompt/visual_title/visual_keywords to narration_broll items."""
    plans = DEEPFAKE_VISUAL_ASSET_PLANS if is_deepfake_topic else GENERIC_VISUAL_ASSET_PLANS
    result = []
    for item in final_timeline:
        item = dict(item)
        t = item.get("type") or item.get("segment_type") or ""
        if t == "narration_broll" and not item.get("visual_mode"):
            sid = item.get("section_id") or ""
            plan = plans.get(sid)
            if plan:
                item["visual_mode"] = plan["visual_mode"]
                item["visual_asset_type"] = plan["visual_asset_type"]
                item["visual_title"] = plan.get("visual_title", "")
                item["visual_asset_prompt"] = plan.get("visual_asset_prompt", "")
                item["visual_keywords"] = plan.get("visual_keywords", [])
            else:
                item["visual_mode"] = "generated_asset"
                item["visual_asset_type"] = "concept_card"
                item["visual_title"] = sid.replace("_", " ").title() if sid else "Section Visual"
                item["visual_asset_prompt"] = ""
                item["visual_keywords"] = []
        result.append(item)
    return result


def _build_repair_prompt(
    current_timeline: list,
    quality_gates: dict,
    narrative_sections: list,
    user_prompt: str,
    iteration: int,
    target_duration: int = 120,
) -> str:
    """Build a structured LLM repair prompt describing failing gates and repair rules."""
    failed_gates = [k for k, v in quality_gates.items() if isinstance(v, bool) and not v]
    offending = quality_gates.get("offending_items", {})

    n_sections = max(1, len(narrative_sections))
    per_section_s = max(8, (target_duration - 10) // n_sections)
    target_words = max(40, int(per_section_s * 2.5))

    gate_descriptions = {
        "all_sections_covered": "Some narrative sections are missing from the timeline.",
        "all_speaker_labels_present": "Some speaker clips have no speaker label.",
        "all_speaker_quotes_complete": "Some speaker clips have incomplete quotes — demote them to narration_broll.",
        "no_instruction_leakage_in_voiceover": "Some voiceover texts start with instruction phrases instead of viewer-facing narration.",
        "all_voiceovers_topic_specific": "Some voiceover texts are generic boilerplate, not specific to the documentary topic.",
        "final_timeline_approved": "The overall timeline does not pass minimum quality standards.",
        "all_narration_broll_have_visual_asset_prompt": "Some narration_broll items are missing visual_asset_prompt.",
    }

    failure_lines = []
    for k in failed_gates:
        desc = gate_descriptions.get(k, "failed")
        off = offending.get(k, [])
        failure_lines.append(f"- {k}: {desc}" + (f" Offending: {off}" if off else ""))
    failure_report = "\n".join(failure_lines)

    sections_json = json.dumps(
        [{"section_id": s["section_id"], "title": s["title"], "purpose": s["purpose"]} for s in narrative_sections],
        indent=2,
    )

    current_tl_summary = json.dumps(
        [
            {
                "type": item.get("type") or item.get("segment_type"),
                "section_id": item.get("section_id"),
                "fragment_id": (item.get("fragment_id") or "")[:50],
                "speaker": item.get("speaker", ""),
                "quote_complete": item.get("quote_complete"),
                "quote_text": (item.get("quote_text") or "")[:100],
                "voiceover_text": (item.get("voiceover_text") or "")[:100],
            }
            for item in current_timeline
            if (item.get("type") or item.get("segment_type")) not in ("title_card", "closing_card")
        ],
        indent=2,
    )

    return (
        f"The validated timeline (iteration {iteration}) failed these quality gates:\n"
        f"{failure_report}\n\n"
        f'ORIGINAL USER PROMPT: "{user_prompt}"\n'
        f"TARGET DURATION: {target_duration} seconds\n\n"
        f"REQUIRED NARRATIVE SECTIONS (all must be present):\n{sections_json}\n\n"
        f"CURRENT TIMELINE:\n{current_tl_summary}\n\n"
        "REPAIR RULES:\n"
        "1. Demote any speaker_clip with quote_complete=false to narration_broll.\n"
        "2. Ensure all narrative sections are present.\n"
        "3. Every narration_broll voiceover_text must be viewer-facing narration — never start with\n"
        "   'Explain', 'Mention', 'Show how', 'End with', 'Start with', 'Open with', 'Describe', 'Close with', etc.\n"
        f"4. Every narration_broll voiceover_text must be ~{target_words} words ({target_words//15}-{target_words//15+2} sentences).\n"
        "5. Every narration_broll must include a non-empty visual_asset_prompt string.\n"
        "6. Keep the documentary order from the REQUIRED NARRATIVE SECTIONS.\n"
        "7. The final video must meaningfully answer the original prompt.\n"
        "8. PRESERVE existing voiceover_text if it is long and correct — only rewrite the offending items.\n\n"
        "Return a repaired final_timeline as a JSON array. Include only the array (no extra keys).\n"
        "Example structure:\n"
        '[\n'
        '  {"type": "speaker_clip", "section_id": "S1_HOOK", "fragment_id": "...", "speaker": "...", '
        '"render_start": 0.0, "render_end": 0.0, "quote_text": "...", "quote_complete": true},\n'
        '  {"type": "narration_broll", "section_id": "S2_CONTEXT", "fragment_id": "...", '
        '"voiceover_text": "evidence-driven narration about the topic...", "visual_asset_prompt": "...", '
        '"visual_title": "Context: Key Concepts", "visual_asset_type": "diagram"}\n'
        "]\n"
    )


def _llm_repair_timeline(
    current_timeline: list,
    quality_gates: dict,
    narrative_sections: list,
    user_prompt: str,
    iteration: int,
    target_duration: int = 120,
) -> "list | None":
    """Ask LLM to repair a failed timeline. Returns new timeline list or None on failure."""
    prompt = _build_repair_prompt(
        current_timeline, quality_gates, narrative_sections, user_prompt, iteration, target_duration
    )
    try:
        raw = post_with_retries({
            "model": STRONG_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 10000},
        })
        repaired = extract_json_array(raw)
        if repaired and isinstance(repaired, list):
            return repaired
        obj = extract_json_object(raw)
        if obj and isinstance(obj.get("final_timeline"), list):
            return obj["final_timeline"]
        log("  [repair] LLM returned no valid JSON array.")
        return None
    except Exception as exc:
        log(f"  [repair] LLM repair call failed: {exc}")
        return None


def check_quality_gates(
    final_timeline: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    """Return (warnings_list, editorial_quality_gates_dict)."""
    warnings: list[str] = []
    offending_items: dict[str, list[str]] = {}

    def _flag(gate_key: str, item_id: str, msg: str) -> None:
        warnings.append(msg)
        offending_items.setdefault(gate_key, []).append(item_id)

    speaker_clips = [x for x in final_timeline if x.get("type") == "speaker_clip"]
    narration_brolls = [x for x in final_timeline if x.get("type") == "narration_broll"]
    all_section_ids = {x.get("section_id") for x in final_timeline if x.get("section_id")}
    required_section_ids = {s["section_id"] for s in narrative_sections}

    # 1. All sections covered
    missing_sids = required_section_ids - all_section_ids
    for sid in sorted(missing_sids):
        _flag("all_sections_covered", sid, f"GATE FAIL: Section {sid} missing from final timeline.")
    gate_sections = not bool(missing_sids)

    # 2. All speaker labels present
    gate_speakers = True
    for sc in speaker_clips:
        fid = (sc.get("fragment_id") or "?")[:40]
        speaker = sc.get("speaker") or ""
        if not speaker:
            _flag("all_speaker_labels_present", fid, f"GATE FAIL: speaker_clip {fid!r} has no speaker label.")
            gate_speakers = False

    # 3. All speaker quotes complete
    gate_quotes = True
    for sc in speaker_clips:
        fid = (sc.get("fragment_id") or "?")[:40]
        if not sc.get("quote_complete"):
            note = sc.get("quote_completeness_note") or "quote_complete=false"
            _flag("all_speaker_quotes_complete", fid, f"GATE WARN: {fid!r} — {note}")
            gate_quotes = False

    # 4. No instruction leakage in voiceover
    instruction_verbs = frozenset(["end with", "explain", "mention", "show how",
                                    "this section", "the video should", "use this clip",
                                    "the narrative should", "start with", "begin with",
                                    "open with", "close with", "describe"])
    gate_no_leak = True
    for nb in narration_brolls:
        sid = nb.get("section_id") or "?"
        vo = (nb.get("voiceover_text") or "").lower()
        for phrase in instruction_verbs:
            if vo.startswith(phrase):
                _flag("no_instruction_leakage_in_voiceover", sid,
                      f"GATE WARN: narration_broll {sid} voiceover starts with instruction verb: {phrase!r}")
                gate_no_leak = False
                break

    # 5. All voiceovers topic-specific (not generic boilerplate)
    generic_phrases = [
        "conversations about artificial intelligence",
        "raising a central question",
        "across the world,",
        "in communities around the world",
        "the following evidence raises",
        "understanding the issue requires",
    ]
    gate_specific = True
    for nb in narration_brolls:
        sid = nb.get("section_id") or "?"
        vo = (nb.get("voiceover_text") or "").lower()
        if not vo:
            _flag("all_voiceovers_topic_specific", sid,
                  f"GATE FAIL: narration_broll {sid} has empty voiceover_text.")
            gate_specific = False
        else:
            for phrase in generic_phrases:
                if phrase in vo:
                    _flag("all_voiceovers_topic_specific", sid,
                          f"GATE WARN: narration_broll {sid} contains generic phrase: {phrase!r}")
                    gate_specific = False
                    break

    # 6. Final timeline approved overall
    gate_approved = gate_sections and gate_speakers and len(speaker_clips) >= 2

    # 7. All narration_broll items have visual_asset_prompt
    gate_visual = True
    for nb in narration_brolls:
        sid = nb.get("section_id") or "?"
        if not nb.get("visual_asset_prompt"):
            _flag("all_narration_broll_have_visual_asset_prompt", sid,
                  f"GATE WARN: narration_broll {sid} missing visual_asset_prompt.")
            gate_visual = False

    gates: dict[str, Any] = {
        "all_sections_covered": gate_sections,
        "all_speaker_labels_present": gate_speakers,
        "all_speaker_quotes_complete": gate_quotes,
        "no_instruction_leakage_in_voiceover": gate_no_leak,
        "all_voiceovers_topic_specific": gate_specific,
        "final_timeline_approved": gate_approved,
        "all_narration_broll_have_visual_asset_prompt": gate_visual,
        "speaker_clip_count": len(speaker_clips),
        "narration_broll_count": len(narration_brolls),
        "sections_covered": sorted(all_section_ids - {None}),
        "offending_items": offending_items,
    }

    return warnings, gates

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM editorial validation layer for AI4IA storylines.")
    parser.add_argument("--storyline", required=True, help="Path to input storyline JSON.")
    parser.add_argument(
        "--transcripts-dir",
        default=str(PROJECT_ROOT / "data" / "transcripts_cleaned_aligned"),
        help="Primary transcripts directory.",
    )
    parser.add_argument(
        "--fallback-transcripts-dir",
        default=str(PROJECT_ROOT / "data" / "transcripts"),
        help="Fallback transcripts directory.",
    )
    parser.add_argument("--output", required=True, help="Path to write validated storyline JSON.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM calls, use score-based fallback for all decisions.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_VALIDATION_ITERATIONS,
        help="Maximum LLM repair iterations for the narrative validation loop.",
    )
    args = parser.parse_args()

    storyline_path = Path(args.storyline).resolve()
    transcripts_dir = Path(args.transcripts_dir).resolve()
    fallback_dir = Path(args.fallback_transcripts_dir).resolve()
    output_path = Path(args.output).resolve()

    log("Starting Step 07b: LLM Editorial Validation")
    log(f"Input:  {storyline_path}")
    log(f"Output: {output_path}")

    # Load storyline
    data = json.loads(storyline_path.read_text(encoding="utf-8"))
    user_prompt = data.get("user_prompt") or ""
    target_duration = int(data.get("target_duration_seconds") or 120)
    prompt_profile = data.get("prompt_profile") or data.get("prompt_intent") or {}
    topic_terms = prompt_profile.get("topic_terms") or []

    export_timeline = data.get("export_timeline") or []
    selected_candidates = data.get("selected_candidates") or []

    # Build candidate lookup by fragment_id
    candidate_map: dict[str, dict[str, Any]] = {}
    for c in selected_candidates:
        fid = c.get("fragment_id")
        if fid:
            candidate_map[str(fid)] = c

    # Merge export_timeline items with candidate data
    fragment_items = [
        item for item in export_timeline
        if item.get("fragment_id") and item.get("type") not in ("title_card", "closing_card")
    ]

    # Detect narrative structure
    narrative_sections = detect_narrative_sections(user_prompt, topic_terms)
    log(f"Narrative structure: {[s['section_id'] for s in narrative_sections]}")
    log(f"Fragment items to validate: {len(fragment_items)}")

    # Validate each fragment
    validated_candidates: list[dict[str, Any]] = []
    llm_fallback_count = 0

    for i, item in enumerate(fragment_items):
        fid = str(item.get("fragment_id") or "")
        source_video = item.get("source_video") or ""
        start = float(item.get("start") or 0)
        end = float(item.get("end") or start + 10)

        log(f"\n[{i+1}/{len(fragment_items)}] Validating: {fid[:60]!r}")
        log(f"  source_video: {source_video[:60]!r}  window: {start:.1f}→{end:.1f}s")

        # Merge candidate data into item
        cand_data = candidate_map.get(fid, {})
        merged = {**cand_data, **item}
        merged["fragment_id"] = fid

        # Load transcript and extract context window
        transcript_path = find_transcript_file(source_video, transcripts_dir, fallback_dir)
        context_window: list[dict[str, Any]] = []

        if transcript_path:
            log(f"  transcript: {transcript_path.name}")
            segs = load_transcript_segments(transcript_path)
            context_window = extract_context_window(segs, start, end)
            log(f"  context window: {len(context_window)} segments ({len(context_window)} around {start:.1f}–{end:.1f}s)")
        else:
            log(f"  transcript: NOT FOUND for {source_video[:50]!r}")

        # Validate
        if args.skip_llm:
            result = fallback_validation(merged)
            log(f"  fallback decision: {result['decision']} | approved={result['approved']}")
        else:
            result = validate_candidate_with_llm(
                candidate=merged,
                context_window=context_window,
                narrative_sections=narrative_sections,
                user_prompt=user_prompt,
                target_duration=target_duration,
            )
            if result is None:
                llm_fallback_count += 1
                log(f"  WARNING: LLM validation failed — using score-based fallback (#{llm_fallback_count}).")
                result = fallback_validation(merged)
                result["_fallback_reason"] = "llm_failed"

            log(f"  LLM decision: {result.get('decision')} | approved={result.get('approved')} | section={result.get('section_id')} | confidence={result.get('confidence', 0):.2f}")
            if not result.get("approved"):
                log(f"  reason: {result.get('reason', '')}")

        # Carry forward source_video from item
        result["source_video"] = source_video
        result["start"] = start
        result["end"] = end
        result["text"] = merged.get("text") or merged.get("moment_text") or ""
        validated_candidates.append(result)

    approved_count = sum(1 for c in validated_candidates if c.get("approved"))
    rejected_count = len(validated_candidates) - approved_count
    log(f"\nValidation complete: {approved_count} approved, {rejected_count} rejected out of {len(validated_candidates)}")
    if llm_fallback_count:
        log(f"WARNING: {llm_fallback_count}/{len(validated_candidates)} fragments used score-based fallback due to LLM failures — results may be lower quality.")

    # Full narrative assembly
    log("\nRunning full narrative assembly...")
    assembled: dict[str, Any] | None = None

    if not args.skip_llm:
        assembled = assemble_narrative_with_llm(
            validated_candidates=validated_candidates,
            narrative_sections=narrative_sections,
            user_prompt=user_prompt,
            target_duration=target_duration,
            storyline_data=data,
        )

    if assembled is None:
        log("  LLM assembly unavailable — using score-based fallback.")
        assembled = build_fallback_final_timeline(
            validated_candidates=validated_candidates,
            narrative_sections=narrative_sections,
            storyline_data=data,
            target_duration=target_duration,
        )

    # Detect topic for defaults — only match actual deepfake terms in the prompt,
    # never match topic_terms (they come from the prompt and would always match)
    is_deepfake_topic = any(
        term.lower() in user_prompt.lower()
        for term in DEEPFAKE_TOPIC_TERMS
    )

    # Build validated_map keyed by fragment_id for speaker resolution
    validated_map: dict[str, dict[str, Any]] = {}
    for vc in validated_candidates:
        fid = vc.get("fragment_id")
        if fid:
            validated_map[str(fid)] = vc

    # Enrich final_timeline with required fields
    raw_timeline = assembled.get("final_timeline") or []
    if not raw_timeline:
        log("  WARNING: assembled final_timeline is empty — using fallback.")
        assembled = build_fallback_final_timeline(
            validated_candidates=validated_candidates,
            narrative_sections=narrative_sections,
            storyline_data=data,
            target_duration=target_duration,
        )
        raw_timeline = assembled.get("final_timeline") or []

    enriched_timeline = enrich_final_timeline(
        final_timeline=raw_timeline,
        candidate_map=candidate_map,
        original_timeline=export_timeline,
        validated_map=validated_map,
        is_deepfake_topic=is_deepfake_topic,
    )

    # Assign fragments to narration_broll items that have no fragment_id
    log("\nAssigning fragments to empty narration_broll items...")
    enriched_timeline = assign_fragments_to_empty_narration_brolls(
        final_timeline=enriched_timeline,
        validated_candidates=validated_candidates,
        original_timeline=export_timeline,
    )

    # Generate voiceover text for narration_broll items with empty voiceover_text
    if not args.skip_llm:
        log("\nGenerating voiceover texts for narration_broll items...")
        enriched_timeline = generate_voiceovers_for_timeline(
            final_timeline=enriched_timeline,
            narrative_sections=narrative_sections,
            user_prompt=user_prompt,
            validated_candidates=validated_candidates,
            target_duration=target_duration,
            storyline_data=data,
        )
    else:
        defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
        for item in enriched_timeline:
            t = item.get("type") or item.get("segment_type") or ""
            if t == "narration_broll" and not (item.get("voiceover_text") or "").strip():
                sid = item.get("section_id") or ""
                item["voiceover_text"] = defaults.get(sid, "")
                item["subtitle_text"] = item["voiceover_text"]

    # Ensure all narrative sections are represented in the timeline
    enriched_timeline = _fill_missing_sections(
        final_timeline=enriched_timeline,
        narrative_sections=narrative_sections,
        validated_candidates=validated_candidates,
        original_timeline=export_timeline,
    )

    # Validate speaker quote completeness
    enriched_timeline = _validate_speaker_quotes(enriched_timeline)

    # Clean remaining instruction leakage from voiceover texts
    defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
    for item in enriched_timeline:
        t = item.get("type") or item.get("segment_type") or ""
        if t == "narration_broll":
            sid = item.get("section_id") or ""
            raw_vo = item.get("voiceover_text") or ""
            cleaned = _clean_voiceover_text(raw_vo, sid, is_deepfake_topic)
            if not cleaned:
                cleaned = defaults.get(sid, "")
            item["voiceover_text"] = cleaned
            item["subtitle_text"] = cleaned

    # Enforce section rhythm: S1 → speaker, all other sections → narration then speaker
    enriched_timeline = _enforce_section_rhythm(enriched_timeline, narrative_sections)

    # Add visual asset plan to all narration_broll items
    enriched_timeline = _add_visual_asset_plan(enriched_timeline, narrative_sections, is_deepfake_topic)

    assembled["final_timeline"] = enriched_timeline

    # Iterative quality-gate repair loop
    max_iters = args.max_iterations
    warnings, quality_gates = check_quality_gates(enriched_timeline, narrative_sections)
    repair_iter = 0
    for repair_iter in range(1, max_iters + 1):
        if _critical_gates_pass(quality_gates):
            log(f"\nAll critical gates pass (iteration {repair_iter}).")
            break
        if args.skip_llm:
            log(f"\nLLM skipped — not repairing critical gate failures.")
            break
        if repair_iter >= max_iters:
            log(f"\nMax repair iterations ({max_iters}) reached. Proceeding with current timeline.")
            break
        failed = [k for k, v in quality_gates.items() if isinstance(v, bool) and not v and k in CRITICAL_GATE_NAMES]
        log(f"\nCritical gate failures on iteration {repair_iter}: {failed}")
        log("Sending repair prompt to LLM...")
        repaired_tl = _llm_repair_timeline(
            current_timeline=enriched_timeline,
            quality_gates=quality_gates,
            narrative_sections=narrative_sections,
            user_prompt=user_prompt,
            iteration=repair_iter,
            target_duration=target_duration,
        )
        if not repaired_tl:
            log("  [repair] LLM returned nothing — keeping current timeline.")
            break
        log(f"  [repair] LLM returned {len(repaired_tl)} items — re-enriching...")
        enriched_timeline = enrich_final_timeline(
            final_timeline=repaired_tl,
            candidate_map=candidate_map,
            original_timeline=export_timeline,
            validated_map=validated_map,
            is_deepfake_topic=is_deepfake_topic,
        )
        enriched_timeline = assign_fragments_to_empty_narration_brolls(
            enriched_timeline, validated_candidates, export_timeline
        )
        if not args.skip_llm:
            enriched_timeline = generate_voiceovers_for_timeline(
                enriched_timeline, narrative_sections, user_prompt, validated_candidates,
                target_duration=target_duration, storyline_data=data,
            )
        enriched_timeline = _fill_missing_sections(
            enriched_timeline, narrative_sections, validated_candidates, export_timeline
        )
        enriched_timeline = _validate_speaker_quotes(enriched_timeline)
        _defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
        for _item in enriched_timeline:
            _t = _item.get("type") or _item.get("segment_type") or ""
            if _t == "narration_broll":
                _sid = _item.get("section_id") or ""
                _raw_vo = _item.get("voiceover_text") or ""
                _cleaned = _clean_voiceover_text(_raw_vo, _sid, is_deepfake_topic)
                if not _cleaned:
                    _cleaned = _defaults.get(_sid, "")
                _item["voiceover_text"] = _cleaned
                _item["subtitle_text"] = _cleaned
        enriched_timeline = _enforce_section_rhythm(enriched_timeline, narrative_sections)
        enriched_timeline = _add_visual_asset_plan(enriched_timeline, narrative_sections, is_deepfake_topic)
        warnings, quality_gates = check_quality_gates(enriched_timeline, narrative_sections)
    else:
        repair_iter = max_iters

    if warnings:
        log("\nQuality gate results:")
        for w in warnings:
            log(f"  {w}")
    else:
        log("\nQuality gates: all passed.")
    assembled.setdefault("warnings", [])
    assembled["warnings"].extend(warnings)

    # Print timeline summary
    log("\n=== Validated Final Timeline ===")
    sc_count = quality_gates["speaker_clip_count"]
    nb_count = quality_gates["narration_broll_count"]
    log(f"  total items:     {len(enriched_timeline)}")
    log(f"  speaker_clips:   {sc_count}")
    log(f"  narration_broll: {nb_count}")
    log(f"  quality_score:   {assembled.get('editorial_quality_score', 0):.2f}")
    log(f"  gates: sections={quality_gates['all_sections_covered']}  "
        f"speakers={quality_gates['all_speaker_labels_present']}  "
        f"quotes={quality_gates['all_speaker_quotes_complete']}  "
        f"voiceover_clean={quality_gates['no_instruction_leakage_in_voiceover']}  "
        f"approved={quality_gates['final_timeline_approved']}")

    for i, item in enumerate(enriched_timeline, 1):
        t = item.get("type") or "?"
        sid = item.get("section_id") or ""
        fid = (item.get("fragment_id") or "")[:50]
        speaker = item.get("speaker") or ""
        rs = item.get("render_start") or item.get("start") or ""
        re_ = item.get("render_end") or item.get("end") or ""
        qc = item.get("quote_complete")
        log(f"  [{i}] {t:<18} {sid:<14} {fid}")
        if t == "speaker_clip":
            log(f"       speaker={speaker!r}  {rs}→{re_}  quote_complete={qc}")
            text = (item.get("quote_text") or item.get("subtitle_text") or "")[:120]
            log(f"       quote: {text!r}")
        elif t == "narration_broll":
            vo = (item.get("voiceover_text") or "")[:120]
            log(f"       vo: {vo!r}")

    # Write output — copy original data, add validated fields
    output_data = dict(data)
    output_data["llm_validated"] = True
    output_data["validation_run_at"] = datetime.utcnow().isoformat()
    output_data["validated_with"] = STRONG_LLM_MODEL if not args.skip_llm else "fallback"
    output_data["repair_iterations"] = repair_iter
    output_data["final_timeline"] = enriched_timeline
    output_data["editorial_quality_gates"] = quality_gates
    output_data["validated_narrative"] = {
        "title": assembled.get("title"),
        "logline": assembled.get("logline"),
        "narrative_arc": assembled.get("narrative_arc") or [],
        "final_timeline": enriched_timeline,
        "rejected_candidates": assembled.get("rejected_candidates") or [],
        "editorial_quality_score": assembled.get("editorial_quality_score", 0),
        "editorial_quality_gates": quality_gates,
        "warnings": assembled.get("warnings") or [],
    }
    # Also merge into storyline sub-object so Step 08 can find it either way
    output_data["storyline"]["final_timeline"] = enriched_timeline
    output_data["storyline"]["validated_narrative"] = output_data["validated_narrative"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nValidated storyline written: {output_path}")
    log("Step 07b complete.")


if __name__ == "__main__":
    main()
