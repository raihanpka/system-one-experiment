"""Report tables: the preface plus each slice's own tables."""
from __future__ import annotations

from ..mapping.tables import mapping_tables
from ..stability.tables import stability_tables
from ..transform.tables import transform_tables

PREFACE = ("All numbers are percentages unless noted. Levels: 0 identical, 1 style only, 2 abbreviation, 3 synonym, "
           "4 Indonesian, 5 legacy codes (OBJ_INFO_ADDR), 6 mixed + decoys. Few cases per level: read as indications.")


def build_tables(store, cases, conds, only=None) -> str:
    return "\n".join([PREFACE, "", *mapping_tables(store, cases, conds, only),
                      *transform_tables(store, conds, only), *stability_tables(store, conds, only)])
