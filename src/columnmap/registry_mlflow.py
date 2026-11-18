# Optional MLflow Prompt Registry backend. Importing this module does not import mlflow; the loader is
# resolved on first use, so the core stays standard-library only unless this backend is actually used.
#
#   registry = MlflowRegistry(prefix="columnmap/", version="production")
#   mapper = Mapper(decider, knowledge, registry=registry)
#
# MLflow prompt versions are immutable and aliases are the deployment pointer. A numeric version loads
# prompts:/<prefix><name>/<version>; an alias loads prompts:/<prefix><name>@<alias>; nothing loads @latest.

from __future__ import annotations

from .registry import Prompt


def _load(uri: str):
    import mlflow  # optional dependency: uv sync --extra registry

    return mlflow.genai.load_prompt(uri)


class MlflowRegistry:
    def __init__(
        self, prefix: str = "columnmap/", version: str | None = None, loader=None
    ):
        self._prefix = prefix.rstrip("/") + "/"
        self._version = version
        self._loader = loader or _load

    def uri(self, name: str, version: str | None = None) -> str:
        spec = version or self._version
        base = f"prompts:/{self._prefix}{name}"
        if spec is None:
            return f"{base}@latest"
        return f"{base}/{spec}" if str(spec).isdigit() else f"{base}@{spec}"

    def get(self, name: str, version: str | None = None) -> Prompt:
        prompt = self._loader(self.uri(name, version))
        if not isinstance(prompt.template, str):
            raise ValueError(
                f"MLflow prompt {name!r} is a chat prompt; columnmap needs text prompts"
            )
        return Prompt(
            name=name,
            template=prompt.template,
            version=str(
                getattr(prompt, "version", "") or version or self._version or ""
            )
            or None,
        )
