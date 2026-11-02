# report tables: the preface plus each slice's own tables

from __future__ import annotations

from ..convert.tables import convert_tables, rule_tables
from ..mapping.tables import compare_tables, mapping_tables
from ..stability.tables import stability_tables
from ..transform.tables import transform_tables

PREFACE = ("All numbers are percentages unless noted. Samples, easiest to hardest: easy_identical (names equal the template), "
           "medium_synonym, hard_codes (system codes such as OBJ_INFO_ADDR), hardest_noisy (mixed styles, extra columns, missing sources), "
           "indonesian. One file per sample: read the numbers as indications.")


def build_tables(store, cases, conds, only=None, other=None, section="mapping") -> str:
    return "\n".join([PREFACE, "", *mapping_tables(store, cases, conds, only, section),
                      *rule_tables(store, cases, conds, only, other, section), *compare_tables(store, cases, conds, only),
                      *convert_tables(store, cases, conds, only, section), *transform_tables(store, conds, only),
                      *stability_tables(store, conds, only)])
