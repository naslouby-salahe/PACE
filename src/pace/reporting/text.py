from collections.abc import Iterable

from pace.types import ConfidenceLevel, DisplayText


def slash_list(parts: Iterable[DisplayText]) -> DisplayText:
    return DisplayText(" / ".join(parts))


def semicolon_list(parts: Iterable[DisplayText]) -> DisplayText:
    return DisplayText("; ".join(parts))


def spaced_semicolon_list(parts: Iterable[DisplayText]) -> DisplayText:
    return DisplayText(" ; ".join(parts))


def bootstrap_interval_text(confidence: ConfidenceLevel) -> DisplayText:
    return DisplayText(f"{confidence:.0%} bootstrap interval")


def resampling_rule_text() -> DisplayText:
    return DisplayText("targets resampled within corpus; corpora fixed")
