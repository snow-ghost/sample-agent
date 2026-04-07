from __future__ import annotations

import re

THREAD_TOKEN_RE = re.compile(r"[a-z0-9]+")
THREAD_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "agent",
        "agents",
        "ai",
        "for",
        "how",
        "i",
        "in",
        "of",
        "on",
        "runtime",
        "the",
        "to",
        "use",
        "with",
    }
)


def resolve_capture_bucket(
    bucket_names: list[str],
    preferred_bucket: str | None,
) -> str | None:
    names = [name for name in bucket_names if name]
    if not names:
        return None
    if preferred_bucket:
        target = preferred_bucket.strip().lower()
        for name in names:
            if name.lower() == target:
                return name
        for name in names:
            if name.lower().startswith(target[:6]) or target.startswith(name.lower()[:6]):
                return name
    return names[0]


def build_capture_markdown(source_text: str) -> tuple[str, str, str]:
    title_match = re.search(r"^#\s+(.+?)\s*$", source_text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Captured source"
    captured_on_match = re.search(r"^Captured on:\s*(\d{4}-\d{2}-\d{2})\s*$", source_text, re.MULTILINE)
    captured_on = captured_on_match.group(1) if captured_on_match else ""
    source_url_match = re.search(r"^Source URL:\s*(\S+)\s*$", source_text, re.MULTILINE)
    source_url = source_url_match.group(1).strip() if source_url_match else ""
    raw_text = source_text.split("Raw text:\n", 1)[1].strip() if "Raw text:\n" in source_text else source_text.strip()

    why_keep = "it preserves a concrete external input worth later review, distillation, or comparison"
    capture_text = (
        f"# {title}\n\n"
        f"- **Source URL:** {source_url}\n"
        f"- **Captured for this template on:** {captured_on}\n"
        f"- **Why keep this:** {why_keep}\n\n"
        "## Raw notes\n\n"
        f"- {raw_text.replace(chr(10) + chr(10), chr(10) + '- ')}\n"
    )
    return title, captured_on, capture_text


def extract_capture_note_lines(text: str) -> list[str]:
    body = text.split("Raw text:\n", 1)[1] if "Raw text:\n" in text else text
    lines: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"^(Captured on|Source URL):", line, re.IGNORECASE):
            continue
        if line.startswith(("- ", "* ")):
            line = line[2:].strip()
        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)
    return lines


def derive_capture_card_title(source_title: str) -> str:
    normalized = source_title.strip()
    if not normalized:
        return "Capture review"
    if ":" in normalized:
        return normalized
    return f"Capture: {normalized}"


def derive_capture_title_from_path(path: str) -> str:
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    return stem.replace("-", " ").strip().title() or "Captured snippet"


def build_direct_capture_markdown(
    source_domain: str,
    capture_path: str,
    snippet: str,
) -> tuple[str, str, str]:
    title = derive_capture_title_from_path(capture_path)
    date_match = re.match(r"^/01_capture/[^/]+/(\d{4}-\d{2}-\d{2})__", capture_path)
    captured_on = date_match.group(1) if date_match else ""
    bullet_lines = [line.strip() for line in snippet.splitlines() if line.strip()]
    raw_notes = "\n".join(f"- {line}" for line in bullet_lines)
    capture_text = (
        f"# {title}\n\n"
        f"- **Source URL:** https://{source_domain}\n"
        f"- **Captured for this template on:** {captured_on}\n"
        "- **Why keep this:** it captures a concrete operating pattern or risk signal worth retaining in the repo.\n\n"
        "## Raw notes\n\n"
        f"{raw_notes}\n"
    )
    return title, captured_on, capture_text


def build_generic_capture_card_markdown(
    card_title: str,
    card_date: str,
    capture_path: str,
    snippet: str,
) -> str:
    lines = extract_capture_note_lines(snippet)
    points = lines[:3] or ["The captured snippet is preserved for later distillation."]
    bullet_text = "\n".join(f"- {line.rstrip('.') }." if not line.endswith((".", "!", "?")) else f"- {line}" for line in points)
    return (
        f"# {card_title}\n\n"
        f"- **Source:** [{capture_path}]({capture_path})\n"
        f"- **Date:** {card_date}\n"
        "- **People:** Unknown\n"
        "- **Topics:** captured source, distillation, review notes\n\n"
        "## Key Points\n"
        f"{bullet_text}\n\n"
        "## Why this matters for current work\n"
        "- This capture preserves reusable source material for later review, synthesis, or retrieval.\n"
    )


def choose_thread_name(
    thread_names: list[str],
    text: str,
) -> str | None:
    candidates = [name for name in thread_names if name.endswith(".md") and not name.startswith("_")]
    if not candidates:
        return None
    text_tokens = {
        token
        for token in THREAD_TOKEN_RE.findall(text.lower())
        if token not in THREAD_STOPWORDS and len(token) > 2
    }
    best_name: str | None = None
    best_score = 0
    for name in candidates:
        stem = name[:-3]
        if "__" in stem:
            stem = stem.split("__", 1)[1]
        thread_tokens = {
            token
            for token in THREAD_TOKEN_RE.findall(stem.replace("_", "-").lower())
            if token not in THREAD_STOPWORDS and len(token) > 2
        }
        score = len(text_tokens & thread_tokens)
        if score > best_score:
            best_score = score
            best_name = name
    if best_name is not None and best_score > 0:
        return best_name
    return candidates[0]
