"""The named samples of the experiment: letter -> (case id, how the input columns are named)."""
SAMPLES = {"a": ("L0-000", "identical names"), "b": ("L3-000", "synonyms"), "c": ("L5-000", "legacy codes"),
           "d": ("L6-000", "mixed, decoys"), "e": ("L1-000", "style only"), "f": ("L2-000", "abbreviations"),
           "g": ("L4-000", "Indonesian")}
DEFAULT = "abcd"
LEVEL_NAMES = ["identical", "style only", "abbreviation", "synonym", "Indonesian", "legacy code", "mixed + decoys"]
