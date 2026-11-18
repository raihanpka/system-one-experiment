# Optional graph harness. The library core stays standard-library only; import this package only when
# the graph extra is installed: uv sync --extra graph

try:
    from .graph import (
        Context,
        MappingState,
        Run,
        build_graph,
        resume_graph,
        retry_policy,
        run_graph,
    )
    from .retry import Retrying
except ModuleNotFoundError as error:  # a clear message instead of a bare ImportError
    if (error.name or "").startswith("langgraph"):
        raise ImportError(
            "columnmap.harness needs the graph extra: run `uv sync --extra graph`"
        ) from error
    raise

__all__ = [
    "Context",
    "MappingState",
    "Retrying",
    "Run",
    "build_graph",
    "resume_graph",
    "retry_policy",
    "run_graph",
]
