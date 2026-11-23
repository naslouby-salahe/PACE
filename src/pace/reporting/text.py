from collections.abc import Iterable

from pace.types import DisplayText


def slash_list(parts: Iterable[DisplayText]) -> DisplayText:
    return DisplayText(" / ".join(parts))


def semicolon_list(parts: Iterable[DisplayText]) -> DisplayText:
    return DisplayText("; ".join(parts))


def spaced_semicolon_list(parts: Iterable[DisplayText]) -> DisplayText:
    return DisplayText(" ; ".join(parts))
