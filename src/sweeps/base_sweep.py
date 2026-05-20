"""Abstract base class for sweep runners."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SweepRunner(ABC):
    """Abstract base for hyperparameter sweep execution.

    Each trial must:
    1. Instantiate model + dataset from the trial config.
    2. Run cross-validation.
    3. Return the validation MAE as the objective.
    4. Log trial hyperparameters and objective to wandb.
    """

    @abstractmethod
    def run(self) -> None:
        """Execute the full sweep."""
