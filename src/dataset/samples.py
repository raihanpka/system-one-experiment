"""The named samples of the experiment: letter -> (case id, how the input columns are named)."""
# one word per drift level; it is part of every case id and file name, e.g. "legacy-000"
LEVEL_WORDS = ["identical", "style", "abbreviation", "synonym", "indonesian", "legacy", "mixed"]
LEVEL_NAMES = ["identical", "style only", "abbreviation", "synonym", "Indonesian", "legacy code", "mixed + decoys"]

SAMPLES = {"a": ("identical-000", "identical names"), "b": ("synonym-000", "synonyms"), "c": ("legacy-000", "legacy codes"),
           "d": ("mixed-000", "mixed, decoys"), "e": ("style-000", "style only"), "f": ("abbreviation-000", "abbreviations"),
           "g": ("indonesian-000", "Indonesian")}
DEFAULT = "abcd"


def word(case_id: str) -> str:
    return case_id.split("-")[0]
