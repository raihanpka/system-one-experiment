# columnmap: map the columns of an input table onto the output fields of a target, with the Decisions model.
#
#   knowledge = Knowledge.load("my_knowledge/")          # outputs.json + reference/*.csv; swap the folder to switch data
#   decider = OpenRouterDecider(api_key, model, base_url)
#   mapping = Mapper(decider, knowledge).map(rows)        # rows: list of dicts, one per input row
#   header, converted = mapping.convert(rows)             # deterministic; fields below the fill score stay empty
#
# Everything under columnmap.bench (synthetic data, ground truth, reports, DeepEval) is for evaluating the mapper only.

from .decider import Decider, OpenRouterDecider
from .knowledge import Knowledge, Output
from .mapper import FILL, Fill, Mapper, Mapping
from .pipeline import Pipeline
from .policy import Policy
from .registry import FileRegistry, Prompt
from .registry_mlflow import MlflowRegistry
from .schema import Column, Schema, extract
from .table import read_table

__all__ = [
    "Column",
    "Decider",
    "FILL",
    "FileRegistry",
    "Fill",
    "Knowledge",
    "Mapper",
    "Mapping",
    "MlflowRegistry",
    "OpenRouterDecider",
    "Output",
    "Pipeline",
    "Policy",
    "Prompt",
    "Schema",
    "extract",
    "read_table",
]
