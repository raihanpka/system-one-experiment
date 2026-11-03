# figure checks: title case for titles, sentence case for axis labels and legend text, readable ticks, legends outside the data

from __future__ import annotations

import re

# words kept lower case inside a title (articles, short conjunctions, short prepositions); longer words are capitalised
SMALL = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "nor",
    "so",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "by",
    "as",
    "vs",
    "per",
}
MIN_TICK_SIZE = 8


def title_case(text: str) -> str:
    words = text.split(" ")
    out = []
    for i, w in enumerate(words):
        core = re.sub(r"[^A-Za-z]", "", w)
        if i > 0 and core.lower() in SMALL:
            out.append(w.lower() if core else w)
        else:
            out.append(w[:1].upper() + w[1:] if w[:1].isalpha() else w)
    return " ".join(out)


def _is_acronym(word: str) -> bool:
    core = re.sub(r"[^A-Za-z]", "", word)
    return len(core) > 1 and core.isupper()


def _title_ok(text: str) -> bool:
    if not text:
        return True
    for i, w in enumerate(text.split(" ")):
        core = re.sub(r"[^A-Za-z]", "", w)
        if not core or _is_acronym(w) or core.isdigit():
            continue
        if i > 0 and core.lower() in SMALL:
            if core[0].isupper():
                return False
            continue
        if not core[0].isupper():
            return False
    return True


def _sentence_ok(text: str) -> bool:
    if not text:
        return True
    if not text[0].isupper() and text[0].isalpha():
        return False
    for w in text.split(" ")[1:]:
        core = re.sub(r"[^A-Za-z]", "", w)
        if len(core) > 1 and core[0].isupper() and not core.isupper():
            return False
    return True


# every problem found in a figure, as short messages; an empty list means the figure is readable
def check_figure(fig) -> list[str]:
    issues = []
    for ax in fig.axes:
        title = ax.get_title()
        if title and not _title_ok(title):
            issues.append(f"title not in title case: {title!r}")
        for label in (ax.get_xlabel(), ax.get_ylabel()):
            if label and not _sentence_ok(label):
                issues.append(f"axis label not in sentence case: {label!r}")
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_text() and t.get_fontsize() < MIN_TICK_SIZE:
                issues.append(
                    f"tick label too small ({t.get_fontsize():.0f} pt): {t.get_text()!r}"
                )
        leg = ax.get_legend()
        if leg is not None and leg.get_gid() != "legend-outside":
            issues.append("legend placed inside the axes (may hide data)")
    return issues
