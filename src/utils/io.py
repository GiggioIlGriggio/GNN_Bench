"""File I/O helpers for common neuroimaging data formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import io


def load_mat(path: Path) -> Dict[str, Any]:
    """Load a ``.mat`` file and return its contents as a dict.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Dict[str, Any]

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    return io.loadmat(path)


def load_npy(path: Path) -> np.ndarray:
    """Load a ``.npy`` file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    np.ndarray

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    return np.load(path)


def load_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Parameters
    ----------
    path : Path
    **kwargs
        Forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path, **kwargs)


def save_json(data: dict, path: Path) -> None:
    """Save a dict as a JSON file.

    Parameters
    ----------
    data : dict
    path : Path
    """
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path: Path) -> dict:
    """Load a JSON file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    dict
    """
    with open(path) as f:
        return json.load(f)
