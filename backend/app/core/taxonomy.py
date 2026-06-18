"""Theme/chapter taxonomy — single source of truth for the hierarchy.

Mirrors the THEMES/CHAPTERS definitions in scripts/05_classify_fragments.py.
Themes are the top level (T1-T7); each chapter (Ch 1-Ch 25) belongs to exactly
one theme via `theme_id`. The frontend uses this to render a nested
theme -> chapters tree.
"""

THEMES: dict[str, dict] = {
    "T1": {"name": "AI Ethics, Governance & Policy", "chapters": ["Ch 1", "Ch 2", "Ch 3", "Ch 4"]},
    "T2": {"name": "AI & Information Access / Media / Journalism", "chapters": ["Ch 5", "Ch 6", "Ch 7", "Ch 8"]},
    "T3": {"name": "AI Equity, Inclusion & Social Justice", "chapters": ["Ch 9", "Ch 10", "Ch 11", "Ch 12"]},
    "T4": {"name": "AI in Health, Science & Environment", "chapters": ["Ch 13", "Ch 14", "Ch 15"]},
    "T5": {"name": "Disinformation, Deepfakes & AI Safety", "chapters": ["Ch 16", "Ch 17", "Ch 18"]},
    "T6": {"name": "AI in Education & Creative Industries", "chapters": ["Ch 19", "Ch 20", "Ch 21", "Ch 22"]},
    "T7": {"name": "AI, Law & Human Rights", "chapters": ["Ch 23", "Ch 24", "Ch 25"]},
}

# chapter_id -> (theme_id, display name). Names match the live classified data.
CHAPTERS: dict[str, dict] = {
    "Ch 1": {"theme_id": "T1", "name": "Ch 1: Foundations of AI Ethics & Philosophy"},
    "Ch 2": {"theme_id": "T1", "name": "Ch 2: AI Governance Models & Oversight"},
    "Ch 3": {"theme_id": "T1", "name": "Ch 3: AI Policy, Regulation & Legislation"},
    "Ch 4": {"theme_id": "T1", "name": "Ch 4: AI Economics, Competition & Markets"},
    "Ch 5": {"theme_id": "T2", "name": "Ch 5: Open Data & Information Accessibility"},
    "Ch 6": {"theme_id": "T2", "name": "Ch 6: AI in Journalism & the Media Landscape"},
    "Ch 7": {"theme_id": "T2", "name": "Ch 7: Government Information & Public Services"},
    "Ch 8": {"theme_id": "T2", "name": "Ch 8: Regional & Satellite Conference Events"},
    "Ch 9": {"theme_id": "T3", "name": "Ch 9: AI Equity in the Global South & Developing Nations"},
    "Ch 10": {"theme_id": "T3", "name": "Ch 10: Gender, Race & Diversity in AI"},
    "Ch 11": {"theme_id": "T3", "name": "Ch 11: Disabilities, Assistive Tech & Accessibility"},
    "Ch 12": {"theme_id": "T3", "name": "Ch 12: Youth, Aging & Generational Perspectives"},
    "Ch 13": {"theme_id": "T4", "name": "Ch 13: AI in Healthcare & Medicine"},
    "Ch 14": {"theme_id": "T4", "name": "Ch 14: AI for Environment, Climate & Sustainability"},
    "Ch 15": {"theme_id": "T4", "name": "Ch 15: AI in Agriculture & Food Security"},
    "Ch 16": {"theme_id": "T5", "name": "Ch 16: Deepfakes, Synthetic Media & Copyright"},
    "Ch 17": {"theme_id": "T5", "name": "Ch 17: AI-Driven Disinformation & Propaganda"},
    "Ch 18": {"theme_id": "T5", "name": "Ch 18: AI Safety, Cybersecurity & Warfare"},
    "Ch 19": {"theme_id": "T6", "name": "Ch 19: AI Literacy, Learning & Education Reform"},
    "Ch 20": {"theme_id": "T6", "name": "Ch 20: AI in Creative Arts, Music & Culture"},
    "Ch 21": {"theme_id": "T6", "name": "Ch 21: Indigenous Knowledge, Language & Localized AI"},
    "Ch 22": {"theme_id": "T6", "name": "Ch 22: AI in Academic Research & Knowledge Production"},
    "Ch 23": {"theme_id": "T7", "name": "Ch 23: AI & Legal Systems / Access to Justice"},
    "Ch 24": {"theme_id": "T7", "name": "Ch 24: Privacy, Human Rights & Civil Liberties"},
    "Ch 25": {"theme_id": "T7", "name": "Ch 25: AI & Intellectual Property"},
}


def chapter_id_from_name(name: str) -> str | None:
    """'Ch 12: Youth...' or 'Ch 12' -> 'Ch 12'."""
    if not name:
        return None
    head = name.split(":", 1)[0].strip()
    return head if head in CHAPTERS else None


def theme_id_from_name(name: str) -> str | None:
    if not name:
        return None
    for tid, t in THEMES.items():
        if t["name"] == name:
            return tid
    return None
