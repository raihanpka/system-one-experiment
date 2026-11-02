# the five samples, easiest to hardest: letter -> (case id, what the input column names look like)
# the word in a case id (for example "hard_codes-000") names the sample in every file and folder

from __future__ import annotations

# the keys are the original drift level numbers; the generator seeds its random choices with them, so they must not change
LEVEL_WORDS = {0: "easy_identical", 3: "medium_synonym", 5: "hard_codes", 6: "hardest_noisy", 4: "indonesian"}
SAMPLES = {"a": ("easy_identical-000", "identical names"), "b": ("medium_synonym-000", "synonyms"),
           "c": ("hard_codes-000", "system codes"), "d": ("hardest_noisy-000", "mixed styles, extra columns, missing sources"),
           "e": ("indonesian-000", "Indonesian")}
DEFAULT = "abcd"
# difficulty axis for figures: the levels in the order of the sample letters
RANK = {lv: i for i, lv in enumerate((0, 3, 5, 6, 4))}


def word(case_id: str) -> str:
    return case_id.split("-")[0]
