from __future__ import annotations

import re

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "TG", "UK", "UP", "WB"
}
STANDARD = re.compile(r"^(?P<state>[A-Z]{2})(?P<district>\d{2})(?P<series>[A-Z]{1,3})(?P<number>\d{4})$")
DELHI = re.compile(r"^DL[1-9][A-Z]{1,3}[0-9]{4}$")
# Rule 50(8) allows one or two letters and excludes I and O from the suffix.
BH = re.compile(r"^(?P<year>\d{2})BH(?P<number>\d{4})(?P<series>[A-HJ-NP-Z]{1,2})$")


def validate_indian_format(text: str) -> dict:
    value = text.upper()
    match = STANDARD.fullmatch(value)
    if match:
        return {"plausible": match.group("state") in STATE_CODES, "format": "standard", "state_code_valid": match.group("state") in STATE_CODES}
    if DELHI.fullmatch(value):
        return {"plausible": True, "format": "delhi-single-digit-zone", "state_code_valid": True}
    match = BH.fullmatch(value)
    if match:
        year = int(match.group("year"))
        return {"plausible": 21 <= year <= 99 and match.group("number") != "0000",
                "format": "bharat-series", "state_code_valid": None}
    return {"plausible": False, "format": "unrecognized", "state_code_valid": None}
