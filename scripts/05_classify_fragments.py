import csv
import json
import math
import re
from pathlib import Path
from datetime import datetime
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRAGMENT_DIR = PROJECT_ROOT / "data" / "fragments"
CLASSIFIED_DIR = PROJECT_ROOT / "data" / "fragments_classified"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "youtube" / "fragment_classification_report.csv"
CATALOG_PATH = PROJECT_ROOT / "data" / "metadata" / "youtube" / "classified_fragments_catalog.jsonl"

CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


THEMES = {
    "T1": {
        "name": "AI Ethics, Governance & Policy",
        "chapters": ["Ch 1", "Ch 2", "Ch 3", "Ch 4"],
        "keywords": [
            "ethics", "ethical", "governance", "policy", "regulation", "legislation",
            "oversight", "accountability", "transparency", "responsible ai", "trust",
            "framework", "standards", "principles", "risk", "human centered",
            "fairness", "bias", "algorithmic", "competition", "market", "economy",
            "economic", "sovereign ai", "national ai strategy", "strategy"
        ],
    },
    "T2": {
        "name": "AI & Information Access / Media / Journalism",
        "chapters": ["Ch 5", "Ch 6", "Ch 7", "Ch 8"],
        "keywords": [
            "access to information", "information accessibility", "open data",
            "public access", "journalism", "media", "news", "press", "content",
            "information", "public information", "government information",
            "public services", "administrative", "transparency", "data access",
            "knowledge", "information systems", "satellite event", "conference",
            "regional", "ifap", "unesco", "freedom of information"
        ],
    },
    "T3": {
        "name": "AI Equity, Inclusion & Social Justice",
        "chapters": ["Ch 9", "Ch 10", "Ch 11", "Ch 12"],
        "keywords": [
            "equity", "inclusion", "inclusive", "justice", "social justice",
            "global south", "developing nations", "africa", "caribbean", "youth",
            "young people", "aging", "elderly", "senior citizens", "gender",
            "race", "diversity", "disability", "disabilities", "accessibility",
            "assistive", "screen reader", "voice interface", "sign language",
            "marginalized", "underserved", "rural", "local communities",
            "empowerment", "digital divide", "minorities"
        ],
    },
    "T4": {
        "name": "AI in Health, Science & Environment",
        "chapters": ["Ch 13", "Ch 14", "Ch 15"],
        "keywords": [
            "health", "healthcare", "medicine", "medical", "patient", "disease",
            "diagnosis", "clinical", "hospital", "medicine waste", "environment",
            "climate", "sustainability", "water", "hydrology", "satellite",
            "agriculture", "food security", "farm", "farming", "smallholders",
            "rainfall", "drought", "biodiversity", "pollution", "natural resources",
            "early warning", "environmental", "cholera", "water quality"
        ],
    },
    "T5": {
        "name": "Disinformation, Deepfakes & AI Safety",
        "chapters": ["Ch 16", "Ch 17", "Ch 18"],
        "keywords": [
            "disinformation", "misinformation", "deepfake", "deepfakes",
            "synthetic media", "propaganda", "information integrity", "truth",
            "false", "fake", "trust", "cybersecurity", "cyberwarfare", "warfare",
            "security", "safety", "online harms", "manipulation", "deception",
            "hallucination", "content authenticity", "fact checking", "fact-checking",
            "participatory propaganda", "weaponization"
        ],
    },
    "T6": {
        "name": "AI in Education & Creative Industries",
        "chapters": ["Ch 19", "Ch 20", "Ch 21", "Ch 22"],
        "keywords": [
            "education", "learning", "literacy", "ai literacy", "students",
            "teachers", "curriculum", "assessment", "academic", "research",
            "knowledge production", "university", "school", "creative", "arts",
            "music", "culture", "poetry", "authors", "creativity", "indigenous",
            "language", "localized ai", "local language", "mother tongue",
            "knowledge systems", "cultural", "heritage", "training", "upskilling",
            "reskilling", "prompt engineering"
        ],
    },
    "T7": {
        "name": "AI, Law & Human Rights",
        "chapters": ["Ch 23", "Ch 24", "Ch 25"],
        "keywords": [
            "law", "legal", "justice", "human rights", "civil liberties",
            "privacy", "data protection", "compliance", "intellectual property",
            "copyright", "rights", "access to justice", "legal systems",
            "regulation", "liability", "public trust", "ip", "policy", "freedom",
            "right to information", "right not to know", "neurotechnology",
            "neuro technology"
        ],
    },
}


CHAPTERS = {
    "Ch 1": {
        "theme_id": "T1",
        "name": "Foundations of AI Ethics & Philosophy",
        "keywords": [
            "ethics", "ethical", "philosophy", "values", "moral", "human centered",
            "responsible ai", "principles", "fairness", "bias", "trust",
            "transparency", "accountability", "humanity", "coexistence"
        ],
    },
    "Ch 2": {
        "theme_id": "T1",
        "name": "AI Governance Models & Oversight",
        "keywords": [
            "governance", "oversight", "accountability", "framework", "model",
            "standards", "audit", "responsibility", "platform responsibility",
            "sovereign ai", "national ai strategy", "institutional"
        ],
    },
    "Ch 3": {
        "theme_id": "T1",
        "name": "AI Policy, Regulation & Legislation",
        "keywords": [
            "policy", "regulation", "legislation", "law", "government", "strategy",
            "national ai", "member states", "recommendation", "unesco recommendation",
            "governance", "public policy", "compliance"
        ],
    },
    "Ch 4": {
        "theme_id": "T1",
        "name": "AI Economics, Competition & Markets",
        "keywords": [
            "economy", "economic", "competition", "markets", "jobs", "workforce",
            "labour", "labor", "productivity", "innovation", "business",
            "small business", "sme", "industry", "investment", "employment"
        ],
    },
    "Ch 5": {
        "theme_id": "T2",
        "name": "Open Data & Information Accessibility",
        "keywords": [
            "open data", "data access", "information accessibility", "access to information",
            "public access", "knowledge", "repositories", "data sets", "datasets",
            "information systems", "information access", "archives", "catalogue",
            "cataloguing", "information"
        ],
    },
    "Ch 6": {
        "theme_id": "T2",
        "name": "AI in Journalism & the Media Landscape",
        "keywords": [
            "journalism", "media", "news", "press", "content creation", "broadcast",
            "journalists", "editorial", "media literacy", "information integrity",
            "content authenticity", "fact checking", "fact-checking"
        ],
    },
    "Ch 7": {
        "theme_id": "T2",
        "name": "Government Information & Public Services",
        "keywords": [
            "government information", "public services", "administrative", "public sector",
            "government", "citizens", "public information", "civic", "services",
            "policy access", "public records", "transparency"
        ],
    },
    "Ch 8": {
        "theme_id": "T2",
        "name": "Regional & Satellite Conference Events",
        "keywords": [
            "conference", "satellite event", "regional", "session", "panel",
            "remarks", "welcome", "speaker", "unesco", "ifap", "ai4ia",
            "dialogue", "event", "audience", "moderator", "panelists"
        ],
    },
    "Ch 9": {
        "theme_id": "T3",
        "name": "AI Equity in the Global South & Developing Nations",
        "keywords": [
            "global south", "developing nations", "africa", "caribbean", "latin america",
            "equity", "inclusion", "digital divide", "local capacity", "sovereignty",
            "infrastructure", "underserved", "rural", "communities", "local problems"
        ],
    },
    "Ch 10": {
        "theme_id": "T3",
        "name": "Gender, Race & Diversity in AI",
        "keywords": [
            "gender", "race", "diversity", "women", "girls", "bias", "representation",
            "minorities", "inclusion", "equitable", "fairness", "discrimination",
            "diverse", "african faces"
        ],
    },
    "Ch 11": {
        "theme_id": "T3",
        "name": "Disabilities, Assistive Tech & Accessibility",
        "keywords": [
            "disability", "disabilities", "assistive", "accessibility", "screen reader",
            "voice interface", "sign language", "hearing", "visual impairment",
            "inclusive design", "accessible", "persons with disabilities"
        ],
    },
    "Ch 12": {
        "theme_id": "T3",
        "name": "Youth, Aging & Generational Perspectives",
        "keywords": [
            "youth", "young people", "students", "elderly", "aging", "senior citizens",
            "generation", "generational", "children", "young innovators", "digital literacy",
            "future generations", "intergenerational"
        ],
    },
    "Ch 13": {
        "theme_id": "T4",
        "name": "AI in Healthcare & Medicine",
        "keywords": [
            "health", "healthcare", "medicine", "medical", "patient", "disease",
            "diagnosis", "clinical", "hospital", "medicine waste", "public health",
            "cholera", "doctor", "health systems"
        ],
    },
    "Ch 14": {
        "theme_id": "T4",
        "name": "AI for Environment, Climate & Sustainability",
        "keywords": [
            "environment", "climate", "sustainability", "water", "hydrology",
            "satellite", "rainfall", "drought", "biodiversity", "pollution",
            "water quality", "natural resources", "early warning", "cyclone",
            "sustainable", "environmental"
        ],
    },
    "Ch 15": {
        "theme_id": "T4",
        "name": "AI in Agriculture & Food Security",
        "keywords": [
            "agriculture", "food security", "farm", "farming", "smallholders",
            "crop", "crops", "livestock", "soil", "irrigation", "farmers",
            "agricultural", "drought resistant"
        ],
    },
    "Ch 16": {
        "theme_id": "T5",
        "name": "Deepfakes, Synthetic Media & Copyright",
        "keywords": [
            "deepfake", "deepfakes", "synthetic media", "copyright", "image generation",
            "voice clone", "cloned", "fake video", "ai-generated image",
            "synthetic", "content authenticity"
        ],
    },
    "Ch 17": {
        "theme_id": "T5",
        "name": "AI-Driven Disinformation & Propaganda",
        "keywords": [
            "disinformation", "misinformation", "propaganda", "false information",
            "fake news", "truth", "information integrity", "manipulation",
            "fact checking", "fact-checking", "devaluation of truth", "trust"
        ],
    },
    "Ch 18": {
        "theme_id": "T5",
        "name": "AI Safety, Cybersecurity & Warfare",
        "keywords": [
            "safety", "cybersecurity", "cyberwarfare", "warfare", "security",
            "weaponization", "threat", "risk", "online harms", "attack",
            "defense", "malicious", "harm"
        ],
    },
    "Ch 19": {
        "theme_id": "T6",
        "name": "AI Literacy, Learning & Education Reform",
        "keywords": [
            "ai literacy", "education", "learning", "students", "teachers",
            "curriculum", "assessment", "school", "digital literacy", "training",
            "upskilling", "reskilling", "prompt engineering", "classroom"
        ],
    },
    "Ch 20": {
        "theme_id": "T6",
        "name": "AI in Creative Arts, Music & Culture",
        "keywords": [
            "creative", "arts", "music", "culture", "poetry", "authors", "poets",
            "creativity", "artist", "screenwriter", "storytelling", "creative industries",
            "cultural production"
        ],
    },
    "Ch 21": {
        "theme_id": "T6",
        "name": "Indigenous Knowledge, Language & Localized AI",
        "keywords": [
            "indigenous", "language", "localized ai", "local language", "mother tongue",
            "community knowledge", "cultural awareness", "local data", "translation",
            "shona", "ndebele", "tonga", "knowledge systems", "local context",
            "local repository", "datasets"
        ],
    },
    "Ch 22": {
        "theme_id": "T6",
        "name": "AI in Academic Research & Knowledge Production",
        "keywords": [
            "academic", "research", "knowledge production", "university", "scholarship",
            "publication", "science", "research data", "data repository",
            "lecturer", "institutional repository", "datasets"
        ],
    },
    "Ch 23": {
        "theme_id": "T7",
        "name": "AI & Legal Systems / Access to Justice",
        "keywords": [
            "legal", "law", "justice", "access to justice", "legal systems",
            "compliance", "court", "rights", "legal information", "law and compliance",
            "lawyers", "regulation"
        ],
    },
    "Ch 24": {
        "theme_id": "T7",
        "name": "Privacy, Human Rights & Civil Liberties",
        "keywords": [
            "privacy", "human rights", "civil liberties", "data protection",
            "digital freedom", "freedom", "surveillance", "rights", "right to know",
            "right not to know", "neurotechnology", "protection"
        ],
    },
    "Ch 25": {
        "theme_id": "T7",
        "name": "AI & Intellectual Property",
        "keywords": [
            "intellectual property", "copyright", "public trust", "ip", "authors",
            "ownership", "licensing", "creative rights", "patent", "content rights"
        ],
    },
}


STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "has", "are", "was",
    "were", "you", "your", "our", "not", "but", "can", "will", "they", "their",
    "about", "into", "also", "more", "when", "what", "how", "why", "who", "where",
    "which", "there", "been", "being", "very", "much", "some", "such", "than",
    "then", "them", "these", "those", "would", "could", "should", "just", "like",
    "because", "through", "within", "without", "between", "across", "today", "thank",
    "thanks", "please", "ladies", "gentlemen"
}


IMPORTANT_PHRASES = [
    "artificial intelligence",
    "access to information",
    "information accessibility",
    "open data",
    "global south",
    "digital divide",
    "ai literacy",
    "human rights",
    "public trust",
    "information integrity",
    "local language",
    "data sovereignty",
    "national ai strategy",
    "media literacy",
    "deepfakes",
    "disinformation",
    "misinformation",
    "youth",
    "ethics",
    "governance",
    "policy",
    "regulation",
    "accessibility",
    "education",
    "health",
    "environment",
    "agriculture",
    "law",
    "copyright",
]


def log(message: str) -> None:
    print(message, flush=True)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", text.lower())
    return [word.strip("-'") for word in words if word not in STOPWORDS and len(word) > 2]


def phrase_count(text: str, phrase: str) -> int:
    return len(re.findall(r"\b" + re.escape(phrase.lower()) + r"\b", text.lower()))


def keyword_score(text: str, keywords: list[str]) -> tuple[float, list[str]]:
    text_norm = normalize(text)
    tokens = tokenize(text_norm)
    token_counter = Counter(tokens)

    score = 0.0
    matched = []

    for kw in keywords:
        kw_norm = kw.lower().strip()
        if " " in kw_norm or "-" in kw_norm:
            count = phrase_count(text_norm, kw_norm)
            if count > 0:
                score += count * 3.0
                matched.append(kw)
        else:
            count = token_counter.get(kw_norm, 0)
            if count > 0:
                score += count * 1.0
                matched.append(kw)

    word_count = max(len(tokens), 1)
    normalized_score = score / math.sqrt(word_count)

    return normalized_score, matched


def score_themes(text: str) -> list[dict]:
    raw_scores = []

    for theme_id, theme in THEMES.items():
        score, matched = keyword_score(text, theme["keywords"])
        if score > 0:
            raw_scores.append(
                {
                    "theme_id": theme_id,
                    "theme": theme["name"],
                    "score": score,
                    "matched_keywords": matched[:12],
                }
            )

    raw_scores.sort(key=lambda item: item["score"], reverse=True)

    if not raw_scores:
        return [
            {
                "theme_id": "T2",
                "theme": THEMES["T2"]["name"],
                "score": 0.15,
                "matched_keywords": [],
            }
        ]

    max_score = raw_scores[0]["score"]

    for item in raw_scores:
        item["score"] = round(min(item["score"] / max_score, 1.0), 3)

    return raw_scores[:3]


def score_chapters(text: str) -> list[dict]:
    raw_scores = []

    for chapter_id, chapter in CHAPTERS.items():
        score, matched = keyword_score(text, chapter["keywords"])
        if score > 0:
            theme_id = chapter["theme_id"]
            raw_scores.append(
                {
                    "chapter_id": chapter_id,
                    "chapter": f"{chapter_id}: {chapter['name']}",
                    "theme_id": theme_id,
                    "theme": THEMES[theme_id]["name"],
                    "score": score,
                    "matched_keywords": matched[:12],
                }
            )

    raw_scores.sort(key=lambda item: item["score"], reverse=True)

    if not raw_scores:
        return [
            {
                "chapter_id": "Ch 8",
                "chapter": "Ch 8: Regional & Satellite Conference Events",
                "theme_id": "T2",
                "theme": THEMES["T2"]["name"],
                "score": 0.15,
                "matched_keywords": [],
            }
        ]

    max_score = raw_scores[0]["score"]

    for item in raw_scores:
        item["score"] = round(min(item["score"] / max_score, 1.0), 3)

    return raw_scores[:5]


def extract_keywords(text: str, theme_tags: list[dict], chapter_tags: list[dict]) -> list[str]:
    text_norm = normalize(text)
    found = []

    for phrase in IMPORTANT_PHRASES:
        if phrase_count(text_norm, phrase) > 0:
            found.append(phrase)

    for tag in theme_tags:
        found.extend(tag.get("matched_keywords", []))

    for tag in chapter_tags:
        found.extend(tag.get("matched_keywords", []))

    tokens = tokenize(text_norm)
    frequent = [
        word for word, count in Counter(tokens).most_common(20)
        if count >= 2 and word not in found
    ]

    combined = []
    seen = set()

    for item in found + frequent:
        clean = item.strip().lower()
        if clean and clean not in seen:
            seen.add(clean)
            combined.append(item)

    return combined[:15]


def create_summary(text: str, max_words: int = 36) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean)

    selected = ""

    for sentence in sentences:
        if len(sentence.split()) >= 8:
            selected = sentence
            break

    if not selected:
        selected = clean

    words = selected.split()

    if len(words) <= max_words:
        return selected.strip()

    return " ".join(words[:max_words]).strip() + "..."


def estimate_clarity(text: str, duration: float, word_count: int) -> float:
    if not text.strip():
        return 0.0

    words_per_second = word_count / max(duration, 1)
    avg_sentence_length = word_count / max(len(re.split(r"[.!?]+", text)), 1)

    score = 0.65

    if 1.4 <= words_per_second <= 3.2:
        score += 0.15
    elif words_per_second > 4.0:
        score -= 0.15

    if 10 <= avg_sentence_length <= 35:
        score += 0.1
    elif avg_sentence_length > 50:
        score -= 0.1

    filler_terms = [
        "um", "uh", "you know", "sort of", "kind of", "okay", "right", "basically"
    ]
    filler_count = sum(phrase_count(text, term) for term in filler_terms)

    if filler_count > 6:
        score -= 0.15
    elif filler_count <= 2:
        score += 0.05

    return round(max(0.0, min(score, 1.0)), 3)


def estimate_quote_strength(text: str) -> float:
    text_norm = normalize(text)

    signal_phrases = [
        "we need", "we must", "we should", "i believe", "i think", "the challenge",
        "the problem", "the opportunity", "this is important", "it is important",
        "ai is", "ai can", "ai must", "access to information is", "the future",
        "not just", "only if", "the question is", "let's", "we are"
    ]

    score = 0.35

    for phrase in signal_phrases:
        if phrase_count(text_norm, phrase) > 0:
            score += 0.07

    if "?" in text:
        score += 0.08

    if any(term in text_norm for term in ["important", "crucial", "essential", "challenge", "opportunity"]):
        score += 0.08

    if 35 <= len(text.split()) <= 170:
        score += 0.12

    return round(max(0.0, min(score, 1.0)), 3)


def estimate_narration_support(text: str, chapter_tags: list[dict]) -> float:
    text_norm = normalize(text)

    score = 0.35

    if len(chapter_tags) >= 2:
        score += 0.15

    if any(term in text_norm for term in [
        "context", "broader", "agenda", "foundation", "framework",
        "society", "communities", "future", "development", "access"
    ]):
        score += 0.2

    if any(term in text_norm for term in [
        "for example", "such as", "in other words", "this means", "therefore"
    ]):
        score += 0.1

    return round(max(0.0, min(score, 1.0)), 3)


def determine_clip_usage(text: str, quote_strength: float, clarity_score: float, narration_support_score: float) -> list[str]:
    text_norm = normalize(text)
    usage = []

    if quote_strength >= 0.65 and clarity_score >= 0.55:
        usage.append("speaker_soundbite")

    if narration_support_score >= 0.6:
        usage.append("narration_support")

    if any(term in text_norm for term in [
        "for example", "evidence", "data", "research", "study", "statistics",
        "we have seen", "we saw", "in zimbabwe", "in jamaica", "in africa"
    ]):
        usage.append("evidence_clip")

    if any(term in text_norm for term in [
        "welcome", "conference", "event", "panel", "speaker", "session",
        "ladies and gentlemen", "thank you"
    ]):
        usage.append("context_visual")

    if any(term in text_norm for term in [
        "we need", "we must", "call to action", "recommendations", "future",
        "should", "must ensure", "let us"
    ]):
        usage.append("synthesis_candidate")

    if not usage:
        usage.append("archive_visual")

    return usage[:4]


def classify_fragment(fragment: dict) -> dict:
    text = fragment.get("text", "")
    duration = float(fragment.get("duration", 0) or 0)
    word_count = int(fragment.get("word_count", 0) or 0)

    theme_tags = score_themes(text)
    chapter_tags = score_chapters(text)
    keywords = extract_keywords(text, theme_tags, chapter_tags)
    summary = create_summary(text)

    clarity_score = estimate_clarity(text, duration, word_count)
    quote_strength = estimate_quote_strength(text)
    narration_support_score = estimate_narration_support(text, chapter_tags)
    clip_usage_candidates = determine_clip_usage(
        text=text,
        quote_strength=quote_strength,
        clarity_score=clarity_score,
        narration_support_score=narration_support_score,
    )

    top_theme_score = theme_tags[0]["score"] if theme_tags else 0
    top_chapter_score = chapter_tags[0]["score"] if chapter_tags else 0
    classification_confidence = round((top_theme_score + top_chapter_score + clarity_score) / 3, 3)

    updated = dict(fragment)

    # Remove old fixed placeholders from Step 04 if present.
    updated.pop("theme", None)
    updated.pop("chapter", None)
    updated.pop("storyline_role", None)
    updated.pop("persona_relevance", None)
    updated.pop("classification_confidence", None)

    updated.update(
        {
            "theme_tags": theme_tags,
            "chapter_tags": chapter_tags,
            "primary_theme": theme_tags[0]["theme"] if theme_tags else None,
            "primary_chapter": chapter_tags[0]["chapter"] if chapter_tags else None,
            "keywords": keywords,
            "summary": summary,
            "clip_usage_candidates": clip_usage_candidates,
            "quote_strength": quote_strength,
            "clarity_score": clarity_score,
            "narration_support_score": narration_support_score,
            "classification_confidence": classification_confidence,
            "classification_method": "rule_based_mvp_v1",
            "classified_at": datetime.utcnow().isoformat(),
        }
    )

    return updated


def classify_fragment_file(path: Path) -> tuple[dict, dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    fragments = payload.get("fragments", [])
    classified_fragments = [classify_fragment(fragment) for fragment in fragments]

    classified_payload = dict(payload)
    classified_payload["classification_method"] = "rule_based_mvp_v1"
    classified_payload["classified_at"] = datetime.utcnow().isoformat()
    classified_payload["fragments"] = classified_fragments

    theme_counter = Counter()
    chapter_counter = Counter()
    usage_counter = Counter()

    for fragment in classified_fragments:
        if fragment.get("primary_theme"):
            theme_counter[fragment["primary_theme"]] += 1
        if fragment.get("primary_chapter"):
            chapter_counter[fragment["primary_chapter"]] += 1
        for usage in fragment.get("clip_usage_candidates", []):
            usage_counter[usage] += 1

    report_row = {
        "fragment_file": path.name,
        "status": "classified",
        "fragment_count": len(classified_fragments),
        "top_theme": theme_counter.most_common(1)[0][0] if theme_counter else "",
        "top_chapter": chapter_counter.most_common(1)[0][0] if chapter_counter else "",
        "speaker_soundbite_count": usage_counter.get("speaker_soundbite", 0),
        "narration_support_count": usage_counter.get("narration_support", 0),
        "evidence_clip_count": usage_counter.get("evidence_clip", 0),
        "context_visual_count": usage_counter.get("context_visual", 0),
        "error": "",
    }

    return classified_payload, report_row


def write_report(rows: list[dict]) -> None:
    fieldnames = [
        "fragment_file",
        "status",
        "fragment_count",
        "top_theme",
        "top_chapter",
        "speaker_soundbite_count",
        "narration_support_count",
        "evidence_clip_count",
        "context_visual_count",
        "error",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log(f"Report written to: {REPORT_PATH}")


def write_catalog(classified_payloads: list[dict]) -> None:
    with CATALOG_PATH.open("w", encoding="utf-8") as f:
        for payload in classified_payloads:
            for fragment in payload.get("fragments", []):
                record = {
                    "fragment_id": fragment.get("fragment_id"),
                    "source_video": fragment.get("source_video"),
                    "video_id": fragment.get("video_id"),
                    "start": fragment.get("start"),
                    "end": fragment.get("end"),
                    "duration": fragment.get("duration"),
                    "word_count": fragment.get("word_count"),
                    "primary_theme": fragment.get("primary_theme"),
                    "primary_chapter": fragment.get("primary_chapter"),
                    "keywords": fragment.get("keywords", []),
                    "summary": fragment.get("summary"),
                    "clip_usage_candidates": fragment.get("clip_usage_candidates", []),
                    "quote_strength": fragment.get("quote_strength"),
                    "clarity_score": fragment.get("clarity_score"),
                    "narration_support_score": fragment.get("narration_support_score"),
                    "classification_confidence": fragment.get("classification_confidence"),
                    "text": fragment.get("text"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log(f"Catalog written to: {CATALOG_PATH}")


def main() -> None:
    log("Starting Step 05: Classify Knowledge Fragments")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Input fragment directory: {FRAGMENT_DIR}")
    log(f"Output classified directory: {CLASSIFIED_DIR}")

    fragment_files = sorted(FRAGMENT_DIR.glob("*.fragments.json"))

    log(f"Found fragment files: {len(fragment_files)}")

    if not fragment_files:
        log("No fragment files found. Run scripts/04_create_fragments.py first.")
        return

    report_rows = []
    classified_payloads = []

    total_fragments = 0

    for path in fragment_files:
        output_path = CLASSIFIED_DIR / path.name.replace(".fragments.json", ".classified.json")

        log(f"CLASSIFYING: {path.name}")

        try:
            classified_payload, report_row = classify_fragment_file(path)

            output_path.write_text(
                json.dumps(classified_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            classified_payloads.append(classified_payload)
            report_rows.append(report_row)
            total_fragments += report_row["fragment_count"]

            log(f"DONE: {path.name} -> {report_row['fragment_count']} fragments classified")

        except Exception as exc:
            log(f"FAILED: {path.name}")
            log(str(exc))

            report_rows.append(
                {
                    "fragment_file": path.name,
                    "status": "failed",
                    "fragment_count": 0,
                    "top_theme": "",
                    "top_chapter": "",
                    "speaker_soundbite_count": 0,
                    "narration_support_count": 0,
                    "evidence_clip_count": 0,
                    "context_visual_count": 0,
                    "error": str(exc),
                }
            )

        write_report(report_rows)

    write_catalog(classified_payloads)

    failed_count = sum(1 for row in report_rows if row["status"] == "failed")

    log("")
    log("Summary")
    log(f"Fragment files processed: {len(report_rows)}")
    log(f"Total fragments classified: {total_fragments}")
    log(f"Failed: {failed_count}")
    log(f"Report: {REPORT_PATH}")
    log(f"Catalog: {CATALOG_PATH}")


if __name__ == "__main__":
    main()
