import re


def normalize_text(raw: str | None) -> str:
    if not raw:
        return ""
    # Strip separators only; preserve every recognized alphanumeric character.
    return re.sub(r"[^A-Z0-9]", "", raw.upper())
