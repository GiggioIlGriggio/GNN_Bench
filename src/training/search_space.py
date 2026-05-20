"""Hyperparameter search-space DSL parser for nested CV (ADR-0008).

Parses the ``params:`` block of ``configs/sweeper/<model>.yaml`` — the exact
same DSL Hydra-Optuna multirun accepts — and produces a list of
:class:`SearchSpec` objects. Each spec knows how to call ``trial.suggest_*``
on an Optuna ``Trial`` and can serialise itself back to the dotted-path
override list the rest of the pipeline expects.

Supported forms (matches Hydra-Optuna semantics):

* ``choice(a, b, c, ...)``   → ``suggest_categorical``
* ``range(low, high)``       → ``suggest_int``  (Python-range, ``[low, high)``)
* ``range(low, high, step)`` → ``suggest_int``  with stride
* ``interval(low, high)``    → ``suggest_float``  (closed)
* ``tag(log, interval(...))``→ ``suggest_float(..., log=True)``
* ``tag(log, range(...))``   → ``suggest_int(..., log=True)``

Bare identifiers inside ``choice(...)`` are read as strings (so
``choice(mean, max, add)`` works without quotes, like in the YAMLs).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import yaml

if TYPE_CHECKING:
    import optuna


_Number = Union[int, float]


@dataclass
class SearchSpec:
    """One parsed search-space entry.

    Attributes
    ----------
    name : str
        Dotted Hydra path, e.g. ``"model.hidden_dim"`` or
        ``"model.model_params.pool_ratio"``.
    kind : str
        One of ``"categorical"``, ``"int"``, ``"float"``.
    choices : Optional[List[Any]]
        Allowed values for ``"categorical"`` specs.
    low : Optional[_Number]
        Lower bound (inclusive) for ``"int"`` and ``"float"``.
    high : Optional[_Number]
        Upper bound (inclusive) for ``"int"`` and ``"float"``.
    step : Optional[int]
        Integer step for ``"int"`` specs.
    log : bool
        Whether to sample on a log scale (``"int"`` / ``"float"``).
    """

    name: str
    kind: str
    choices: Optional[List[Any]] = None
    low: Optional[_Number] = None
    high: Optional[_Number] = None
    step: Optional[int] = None
    log: bool = False

    def suggest(self, trial: "optuna.Trial") -> Any:
        """Sample a value for this spec from an Optuna ``Trial``."""
        if self.kind == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        if self.kind == "int":
            return trial.suggest_int(
                self.name,
                int(self.low),
                int(self.high),
                step=int(self.step or 1),
                log=self.log,
            )
        if self.kind == "float":
            return trial.suggest_float(
                self.name, float(self.low), float(self.high), log=self.log,
            )
        raise ValueError(f"Unknown SearchSpec kind: {self.kind!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_search_space(params: Dict[str, Any]) -> List[SearchSpec]:
    """Parse a Hydra-Optuna ``params`` block into a list of search specs.

    Parameters
    ----------
    params : dict
        Mapping of dotted Hydra paths (e.g. ``"model.hidden_dim"``) to DSL
        expression strings. Non-string values are treated as fixed scalars
        and turned into a single-choice categorical.

    Returns
    -------
    List[SearchSpec]

    Raises
    ------
    ValueError
        On unsupported DSL forms or malformed expressions.
    """
    specs: List[SearchSpec] = []
    for name, expr in params.items():
        if isinstance(expr, str):
            specs.append(_parse_expr(name, expr))
        else:
            specs.append(
                SearchSpec(name=name, kind="categorical", choices=[expr])
            )
    return specs


def load_sweeper_params(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """Read a ``configs/sweeper/<name>.yaml`` and return its ``params:`` block.

    Hydra adds a ``# @package hydra.sweeper`` directive at the top, but the
    body parses as plain YAML — ``yaml.safe_load`` is enough.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    params = data.get("params")
    if params is None:
        raise ValueError(
            f"Sweeper YAML at {yaml_path} has no 'params' block; "
            f"top-level keys are: {sorted(data)}"
        )
    return params


# ---------------------------------------------------------------------------
# Internal: DSL → SearchSpec
# ---------------------------------------------------------------------------


def _parse_expr(name: str, expr: str) -> SearchSpec:
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError as e:
        raise ValueError(
            f"Cannot parse search-space expr {expr!r} for {name!r}: {e}"
        ) from e
    return _interpret(name, node, log=False)


def _interpret(name: str, node: ast.AST, log: bool) -> SearchSpec:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(
            f"Search-space expression for {name!r} must be a function call "
            f"(got {ast.dump(node)})"
        )
    func = node.func.id

    if func == "choice":
        choices = [_literal(a) for a in node.args]
        if not choices:
            raise ValueError(f"choice(...) for {name!r} requires >= 1 argument")
        return SearchSpec(name=name, kind="categorical", choices=choices)

    if func == "range":
        if len(node.args) not in (2, 3):
            raise ValueError(
                f"range(...) for {name!r} requires 2 or 3 args, got "
                f"{len(node.args)}"
            )
        low = int(_literal(node.args[0]))
        high_excl = int(_literal(node.args[1]))
        step = int(_literal(node.args[2])) if len(node.args) == 3 else 1
        if step <= 0:
            raise ValueError(f"range(...) step must be positive, got {step}")
        if high_excl <= low:
            raise ValueError(
                f"range(...) requires high > low for {name!r}; got "
                f"low={low} high={high_excl}"
            )
        # Python-range semantics: [low, high) → Optuna closed range
        # whose top inclusive bound is the largest value reachable in stride.
        n_steps = (high_excl - low - 1) // step
        high_incl = low + n_steps * step
        return SearchSpec(
            name=name, kind="int", low=low, high=high_incl, step=step, log=log,
        )

    if func == "interval":
        if len(node.args) != 2:
            raise ValueError(
                f"interval(...) for {name!r} requires 2 args, got {len(node.args)}"
            )
        low = float(_literal(node.args[0]))
        high = float(_literal(node.args[1]))
        if high <= low:
            raise ValueError(
                f"interval(...) requires high > low for {name!r}; got "
                f"low={low} high={high}"
            )
        return SearchSpec(name=name, kind="float", low=low, high=high, log=log)

    if func == "tag":
        # tag(log, <inner>) — only the 'log' tag is supported.
        tag_names: List[str] = []
        inner_calls: List[ast.Call] = []
        for a in node.args:
            if isinstance(a, ast.Name):
                tag_names.append(a.id)
            elif isinstance(a, ast.Call):
                inner_calls.append(a)
            else:
                raise ValueError(
                    f"Unexpected tag(...) argument for {name!r}: {ast.dump(a)}"
                )
        unknown = [t for t in tag_names if t != "log"]
        if unknown:
            raise ValueError(
                f"tag(...) for {name!r}: only 'log' is supported, got {unknown}"
            )
        if "log" not in tag_names:
            raise ValueError(
                f"tag(...) for {name!r} must include the 'log' tag"
            )
        if len(inner_calls) != 1:
            raise ValueError(
                f"tag(...) for {name!r} must wrap exactly one distribution"
            )
        return _interpret(name, inner_calls[0], log=True)

    raise ValueError(
        f"Unknown search-space function {func!r} for {name!r}. "
        f"Expected one of: choice, range, interval, tag."
    )


def _literal(node: ast.AST) -> Any:
    """Resolve an AST node to a Python literal.

    Accepts ``Constant`` (number / string / bool / None), ``Name`` (treated
    as a string identifier — Hydra DSL allows unquoted strings in choices),
    and ``UnaryOp(-, ...)`` (negative numeric literals).
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal(node.operand)
        if isinstance(inner, (int, float)):
            return -inner
    raise ValueError(
        f"Cannot interpret AST node {ast.dump(node)} as a search-space literal"
    )


# ---------------------------------------------------------------------------
# Applying chosen overrides to pydantic config objects
# ---------------------------------------------------------------------------


@dataclass
class TrialOverrides:
    """A flat dict of ``dotted.path → value`` chosen for one trial.

    Apply via :meth:`apply` to produce updated copies of the model and
    trainer configs without mutating the originals.
    """

    values: Dict[str, Any] = field(default_factory=dict)

    def apply(self, *, model_cfg, trainer_cfg):
        """Return ``(new_model_cfg, new_trainer_cfg)`` with overrides applied.

        Paths are dispatched by their first segment:

        * ``model.<field>`` → update the model config.
        * ``model.model_params.<key>`` → update a key inside ``model_params``.
        * ``trainer.<field>`` → update the trainer config.

        Any other prefix raises ``ValueError`` — the DSL doesn't support
        cross-cutting sweeps in this PR.
        """
        model_updates: Dict[str, Any] = {}
        model_params_updates: Dict[str, Any] = {}
        trainer_updates: Dict[str, Any] = {}
        for path, value in self.values.items():
            parts = path.split(".")
            if parts[0] == "model":
                if len(parts) == 2:
                    model_updates[parts[1]] = value
                elif len(parts) == 3 and parts[1] == "model_params":
                    model_params_updates[parts[2]] = value
                else:
                    raise ValueError(
                        f"Unsupported model override path: {path!r} "
                        f"(allowed: model.<field>, model.model_params.<key>)"
                    )
            elif parts[0] == "trainer":
                if len(parts) != 2:
                    raise ValueError(
                        f"Unsupported trainer override path: {path!r} "
                        f"(allowed: trainer.<field>)"
                    )
                trainer_updates[parts[1]] = value
            else:
                raise ValueError(
                    f"Unsupported override prefix {parts[0]!r} in {path!r}; "
                    f"only 'model.' and 'trainer.' are recognised."
                )

        if model_params_updates:
            new_model_params = dict(model_cfg.model_params)
            new_model_params.update(model_params_updates)
            model_updates["model_params"] = new_model_params

        new_model_cfg = (
            model_cfg.model_copy(update=model_updates) if model_updates else model_cfg
        )
        new_trainer_cfg = (
            trainer_cfg.model_copy(update=trainer_updates) if trainer_updates else trainer_cfg
        )
        return new_model_cfg, new_trainer_cfg
