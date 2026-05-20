"""Registry and built-in implementations for composite label operations.

Design
------
Each composite is a class that inherits from :class:`BaseComposite` and is
registered via the :func:`register_composite` decorator.  Composites can be
**stateless** (e.g. weighted sum, IES) — ``fit`` is a no-op — or
**stateful** (e.g. PCA) — ``fit`` learns parameters from training data.

Adding a new composite
~~~~~~~~~~~~~~~~~~~~~~
1. Create a subclass of :class:`BaseComposite`.
2. Decorate it with ``@register_composite("my_name")``.
3. Implement ``fit`` and ``transform``.
4. Use it in config::

       composite_columns: [col_a, col_b]
       composite_method: my_name
       composite_params:
         whatever_param: 42

The constructor receives ``**composite_params`` from the YAML config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_COMPOSITE_REGISTRY: Dict[str, Type["BaseComposite"]] = {}


def register_composite(name: str):
    """Class decorator — register a :class:`BaseComposite` subclass.

    Parameters
    ----------
    name : str
        Unique composite name (used in ``composite_method`` YAML field).
    """
    def wrapper(cls: Type["BaseComposite"]) -> Type["BaseComposite"]:
        if name in _COMPOSITE_REGISTRY:
            raise ValueError(
                f"Composite '{name}' is already registered "
                f"({_COMPOSITE_REGISTRY[name].__qualname__})."
            )
        _COMPOSITE_REGISTRY[name] = cls
        return cls
    return wrapper


def get_composite(
    name: str,
    params: Optional[Dict[str, Any]] = None,
) -> "BaseComposite":
    """Instantiate a registered composite by name.

    Parameters
    ----------
    name : str
        Registered composite name.
    params : dict, optional
        Keyword arguments forwarded to the composite constructor.

    Returns
    -------
    BaseComposite

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in _COMPOSITE_REGISTRY:
        available = list(_COMPOSITE_REGISTRY.keys())
        raise KeyError(
            f"Unknown composite '{name}'. Available: {available}"
        )
    return _COMPOSITE_REGISTRY[name](**(params or {}))


def list_composites() -> List[str]:
    """Return all registered composite names."""
    return list(_COMPOSITE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseComposite(ABC):
    """Abstract base for composite label operations.

    Subclasses compute a single scalar label per subject from K component
    columns.  Stateless composites (most common) leave ``fit`` as a no-op.
    """

    @abstractmethod
    def fit(self, components: np.ndarray) -> None:
        """Fit on training data (no-op for stateless composites).

        Parameters
        ----------
        components : np.ndarray
            Shape ``[N_train, K]``.
        """

    @abstractmethod
    def transform(self, components: np.ndarray) -> np.ndarray:
        """Compute the composite from component columns.

        Parameters
        ----------
        components : np.ndarray
            Shape ``[N, K]`` — one row per subject, one column per component.

        Returns
        -------
        np.ndarray
            Shape ``[N]`` — one scalar label per subject.
        """

    def fit_transform(self, components: np.ndarray) -> np.ndarray:
        """Convenience: ``fit`` then ``transform``."""
        self.fit(components)
        return self.transform(components)


# ---------------------------------------------------------------------------
# Built-in composites
# ---------------------------------------------------------------------------

@register_composite("weighted")
class WeightedComposite(BaseComposite):
    """Weighted sum of component columns.

    Parameters
    ----------
    weights : List[float]
        One weight per column (same order as ``composite_columns``).
    """

    def __init__(self, weights: List[float]) -> None:
        self.weights = np.array(weights, dtype=float)

    def fit(self, components: np.ndarray) -> None:
        pass  # stateless

    def transform(self, components: np.ndarray) -> np.ndarray:
        if components.shape[1] != len(self.weights):
            raise ValueError(
                f"WeightedComposite: expected {len(self.weights)} columns, "
                f"got {components.shape[1]}"
            )
        return (components @ self.weights).astype(float)


@register_composite("ies")
class IESComposite(BaseComposite):
    """Inverse Efficiency Score: RT / Accuracy.

    Expects exactly 2 columns in order: ``[reaction_time, accuracy]``.

    The IES combines speed and accuracy into a single measure where
    higher values indicate worse performance.  Column order matters:
    the **first** column is the numerator (RT) and the **second** is
    the denominator (accuracy, typically proportion correct 0–1).
    """

    def fit(self, components: np.ndarray) -> None:
        pass  # stateless

    def transform(self, components: np.ndarray) -> np.ndarray:
        if components.shape[1] != 2:
            raise ValueError(
                f"IES requires exactly 2 columns [RT, accuracy], "
                f"got {components.shape[1]}"
            )
        rt = components[:, 0]
        accuracy = components[:, 1]
        # Guard against division by zero or near-zero (floating-point and
        # genuine zeros both handled).  1e-6 is well below any real accuracy
        # proportion while keeping IES numerically stable on raw-scale data.
        # NOTE: normalize_components must be False for IES labels — z-scoring
        # accuracy centres it around 0, which would make this guard useless.
        safe_accuracy = np.where(np.abs(accuracy) < 1e-6,
                                 np.sign(accuracy) * 1e-6, accuracy)
        # Treat exact zeros as a small positive value (no meaningful sign).
        safe_accuracy = np.where(safe_accuracy == 0, 1e-6, safe_accuracy)
        return (rt / safe_accuracy).astype(float)


@register_composite("pca")
class PCAComposite(BaseComposite):
    """First principal component of the component columns.

    This is a **stateful** composite — ``fit`` computes the PCA transform
    from training data, and ``transform`` projects onto the first PC.

    Parameters
    ----------
    n_components : int
        Number of PCA components to compute (only the first is returned
        as the scalar label).  Default ``1``.
    """

    def __init__(self, n_components: int = 1) -> None:
        self.n_components = n_components
        self._mean: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None

    def fit(self, components: np.ndarray) -> None:
        self._mean = components.mean(axis=0)
        X_c = components - self._mean
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        self._components = Vt[: self.n_components]

    def transform(self, components: np.ndarray) -> np.ndarray:
        if self._mean is None or self._components is None:
            raise RuntimeError(
                "fit() must be called before transform() for PCA composite"
            )
        X_c = components - self._mean
        return (X_c @ self._components[0]).astype(float)


@register_composite("precision")
class PrecisionComposite(BaseComposite):
    """Signal detection precision: TP / (TP + FP + eps).

    Expects exactly 2 columns in order: ``[TP, FP]``.

    The small epsilon (default ``1e-9``) guards against division by zero
    when both TP and FP are zero.  ``normalize_components`` must be
    ``False`` — z-scoring count columns would destroy the ratio semantics.

    Parameters
    ----------
    eps : float
        Stability epsilon added to the denominator.  Default ``1e-9``.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps

    def fit(self, components: np.ndarray) -> None:
        pass  # stateless

    def transform(self, components: np.ndarray) -> np.ndarray:
        if components.shape[1] != 2:
            raise ValueError(
                f"PrecisionComposite requires exactly 2 columns [TP, FP], "
                f"got {components.shape[1]}"
            )
        tp = components[:, 0]
        fp = components[:, 1]
        return (tp / (tp + fp + self.eps)).astype(float)


@register_composite("ies_precision")
class IESPrecisionComposite(BaseComposite):
    """Inverse Efficiency Score using precision as the accuracy term.

    Computes ``RT / (TP / (TP + FP + eps))``, equivalently
    ``RT * (TP + FP + eps) / TP``.

    Expects exactly 3 columns in order: ``[RT, TP, FP]``.

    This is appropriate for tasks (e.g. PCPT) where accuracy is defined
    as precision (TP / (TP + FP)) rather than as a proportion correct.
    ``normalize_components`` must be ``False``.

    Parameters
    ----------
    eps : float
        Stability epsilon added to the denominator.  Default ``1e-9``.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps

    def fit(self, components: np.ndarray) -> None:
        pass  # stateless

    def transform(self, components: np.ndarray) -> np.ndarray:
        if components.shape[1] != 3:
            raise ValueError(
                f"IESPrecisionComposite requires exactly 3 columns "
                f"[RT, TP, FP], got {components.shape[1]}"
            )
        rt = components[:, 0]
        tp = components[:, 1]
        fp = components[:, 2]
        safe_tp = np.where(np.abs(tp) < self.eps, self.eps, tp)
        return (rt * (tp + fp + self.eps) / safe_tp).astype(float)
