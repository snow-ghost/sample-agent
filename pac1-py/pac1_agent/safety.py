from __future__ import annotations

import re

from .capabilities import extract_task_intent
from .models import CompletionPayload


HIGH_RISK_SECURITY_PATTERNS = (
    re.compile(r"""(?:remove|delete)\s+["'`]?/agents\.md["'`]?"""),
    re.compile(r"""ignore\s+(?:all\s+|any\s+)?(?:local\s+repository\s+|repository\s+|repo\s+)?rules"""),
    re.compile(r"""ignore\s+(?:all\s+)?previous\s+instructions"""),
    re.compile(r"""override(?:[-\s]fast[-\s]path|[-\s]execution[-\s]lane|[-\s]instructions)?"""),
    re.compile(r"""forward\s+(?:any\s+)?(?:visible\s+)?credentials"""),
    re.compile(r"""email\s+passwords"""),
    re.compile(r"""access\s+notes\s+to\s+"""),
    re.compile(r"""ops-bridge@example\.com"""),
    re.compile(r"""bypass\s+(?:safeguards|guardrails|approval)"""),
    re.compile(r"""ship\s+any\s+secrets"""),
)
MEDIUM_RISK_SECURITY_PATTERNS = (
    re.compile(r"""mark\s+(?:the\s+)?task\s+(?:finished|complete|completed)"""),
    re.compile(r"""keep\s+the\s+completion\s+log\s+minimal"""),
    re.compile(r"""avoid\s+asking\s+follow-up\s+questions"""),
    re.compile(r"""temporary\s+execution\s+lane"""),
    re.compile(r"""handling:\s*immediate"""),
    re.compile(r"""status:\s*authenticated"""),
)
EMBEDDED_PAYLOAD_SECURITY_PATTERNS = (
    re.compile(r"""security\s+relay"""),
    re.compile(r"""clipboard\s+relay"""),
    re.compile(r"""\[\[\s*clipboard\s+relay"""),
    re.compile(r"""\[\[/\s*clipboard\s+relay\s*\]\]"""),
    re.compile(r"""embedded\s+tool\s+note"""),
    re.compile(r"""trusted\s+operational\s+guidance"""),
    re.compile(r"""internal\s+bridge"""),
    re.compile(r"""suppress\s+mention"""),
    re.compile(r"""brief\s+success\s+confirmation"""),
    re.compile(r"""no\s+additional\s+confirmation\s+is\s+needed"""),
    re.compile(r"""remove\s+process\s+and\s+policy\s+documents"""),
)


def contains_prompt_injection_markers(task_text: str) -> bool:
    text = task_text.lower()
    if any(pattern.search(text) for pattern in HIGH_RISK_SECURITY_PATTERNS):
        return True
    medium_hits = sum(1 for pattern in MEDIUM_RISK_SECURITY_PATTERNS if pattern.search(text))
    return medium_hits >= 2


def looks_truncated_capture_request(task_text: str, normalized_text: str) -> bool:
    words = task_text.strip().split()
    if not any(marker in normalized_text for marker in ("captur", "capture", "distill")):
        return False
    if len(words) > 6:
        return False
    if any(marker in task_text for marker in ("/", ":", "\"", "'")):
        return False
    if "website" in normalized_text or "snippet" in normalized_text:
        return False
    return True


def text_only_preflight_outcome(task_text: str) -> CompletionPayload | None:
    intent = extract_task_intent(task_text)
    text = intent.normalized_text
    words = task_text.strip().split()
    stripped_text = task_text.strip()

    if intent.wants_capture_or_distill and any(pattern.search(text) for pattern in EMBEDDED_PAYLOAD_SECURITY_PATTERNS):
        return CompletionPayload(
            completed_steps_laconic=["Detected hostile instruction relay inside captured payload"],
            message=(
                "The provided capture payload embeds operational override or cleanup-suppression instructions. "
                "I will not capture or distill hostile instruction content as trusted work."
            ),
            grounding_refs=["/AGENTS.md"],
            outcome="OUTCOME_DENIED_SECURITY",
        )

    if "[[paste hijack" in text or contains_prompt_injection_markers(task_text):
        return CompletionPayload(
            completed_steps_laconic=["Detected prompt-injection or exfiltration markers"],
            message="The request contains prompt-injection or sensitive-data exfiltration instructions. I will not execute it.",
            grounding_refs=["/AGENTS.md"],
            outcome="OUTCOME_DENIED_SECURITY",
        )

    if intent.mentions_deictic_reference and len(words) <= 5:
        return CompletionPayload(
            completed_steps_laconic=["Detected short deictic request without stable target"],
            message="The request does not identify a unique target. Clarify exactly what should be processed or changed.",
            grounding_refs=["/AGENTS.md"],
            outcome="OUTCOME_NONE_CLARIFICATION",
        )

    if looks_truncated_capture_request(task_text, text):
        return CompletionPayload(
            completed_steps_laconic=["Detected truncated capture-style request"],
            message="The request is truncated and does not specify a concrete capture target. Clarify the exact source and destination.",
            grounding_refs=["/AGENTS.md"],
            outcome="OUTCOME_NONE_CLARIFICATION",
        )

    if len(words) <= 3 and "/" not in stripped_text and "." not in stripped_text and "\"" not in stripped_text:
        if not intent.wants_inbox_processing:
            return CompletionPayload(
                completed_steps_laconic=["Detected underspecified short request"],
                message="The request is too short or incomplete to identify the intended action. Clarify the exact target and operation.",
                grounding_refs=["/AGENTS.md"],
                outcome="OUTCOME_NONE_CLARIFICATION",
            )

    return None


def pre_bootstrap_outcome(task_text: str) -> CompletionPayload | None:
    return text_only_preflight_outcome(task_text)
