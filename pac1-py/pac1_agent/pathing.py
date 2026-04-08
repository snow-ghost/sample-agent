from __future__ import annotations

import posixpath
from pathlib import PurePosixPath


AGENT_FILE_NAMES = ("AGENTS.md", "AGENTS.MD")
README_FILE_NAMES = ("README.md", "README.MD")


def normalize_repo_path(path: str) -> str:
    candidate = (path or "").strip().replace("\\", "/")
    if not candidate or candidate == ".":
        return "/"
    candidate = f"/{candidate.lstrip('/')}"
    normalized = posixpath.normpath(candidate)
    return normalized if normalized.startswith("/") else f"/{normalized}"


def candidate_read_paths(path: str) -> list[str]:
    normalized = normalize_repo_path(path)
    path_obj = PurePosixPath(normalized)
    variants = [normalized]

    basename_variants = {
        "agents.md": AGENT_FILE_NAMES,
        "readme.md": README_FILE_NAMES,
    }.get(path_obj.name.lower())
    if basename_variants is None:
        return variants

    for name in basename_variants:
        candidate = normalize_repo_path(str(path_obj.with_name(name)))
        if candidate not in variants:
            variants.append(candidate)
    return variants


def is_agent_instruction_path(path: str) -> bool:
    return PurePosixPath(normalize_repo_path(path)).name.lower() == "agents.md"
