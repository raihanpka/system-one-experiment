# report tables: the preface plus each slice's own tables

from __future__ import annotations

from ..convert.tables import honest_tables, rule_tables
from ..mapping.tables import compare_tables, mapping_tables

PREFACE = ("All numbers are percentages unless noted. Samples, easiest to hardest: easy_identical (names equal the template), "
           "medium_synonym, technical_terms (unusual technical terms such as OBJ_INFO_ADDR), hardest_noisy (mixed styles, extra columns, missing sources). "
           "One file per sample: read the numbers as indications.")


def build_tables(store, cases, conds, only=None, other=None, section="mapping") -> str:
    return "\n".join([PREFACE, "", *honest_tables(store, cases, conds, only), *mapping_tables(store, cases, conds, only, section),
                      *rule_tables(store, cases, conds, only, other, section), *compare_tables(store, cases, conds, only)])
