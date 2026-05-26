"""Regression tests for training selection and logging behavior."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_geometric.data

from src.configs.logging_config import LoggingConfig
from src.configs.trainer_config import TrainerConfig
from src.logging.wandb_logger import WandbLogger
from src.models.base_model import BrainGNN
from src.training.cross_validation import CrossValidator
from src.training.label_normalizer import LabelNormalizer
from src.training.metrics import compute_metrics
from src.training.trainer import Trainer, TrainResult


class _EchoModel(BrainGNN):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode(self, data: object) -> torch.Tensor:
        return data.y.view(-1, 1)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding + self.bias


class _ConstantModel(BrainGNN):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def encode(self, data: object) -> torch.Tensor:
        batch_size = int(data.y.view(-1).shape[0])
        return self.value.expand(batch_size, 1)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding


class _FakeBatch:
    def __init__(self, y_values: list[float]) -> None:
        self.y = torch.tensor(y_values, dtype=torch.float32)
        self.num_graphs = len(y_values)

    def to(self, device: torch.device) -> "_FakeBatch":
        self.y = self.y.to(device)
        return self


class _FakeTrainer:
    def __init__(self) -> None:
        self.logger = MagicMock()

    def fit(self, model, train_loader, val_loader, inverse_transform, fold_idx, on_epoch_end_callback=None):
        with torch.no_grad():
            model.value.fill_(9.0)
        return TrainResult(
            best_epoch=1,
            best_val_metrics={"mae": 2.0, "rmse": 2.0, "r2": 0.0, "pearson_r": 0.0},
            best_model_state_dict={"value": torch.tensor(2.0)},
            last_model_state_dict={"value": torch.tensor(9.0)},
            last_epoch=3,
            last_val_metrics={"mae": 9.0, "rmse": 9.0, "r2": 0.0, "pearson_r": 0.0},
        )

    def predict(self, model, loader, inverse_transform):
        y_true = np.array([float(graph.y.item()) for graph in loader.dataset], dtype=np.float32)
        y_pred = np.full_like(y_true, float(model.value.detach().cpu().item()))
        return y_true, y_pred

    def evaluate(self, model, loader, inverse_transform, split):
        y_true, y_pred = self.predict(model, loader, inverse_transform)
        return compute_metrics(y_true, y_pred)


def test_wandb_logger_logs_explicit_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    wandb = SimpleNamespace(log=MagicMock())
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    logger = WandbLogger(LoggingConfig(enabled=True))
    logger.log_fold_metrics(3, {"r2": 0.42}, split="val", epoch=7)

    logged = wandb.log.call_args.args[0]
    assert logged["fold_3/val/r2"] == pytest.approx(0.42)
    assert logged["fold_3/epoch"] == 7
    assert logged["fold_3/val/epoch"] == 7


def test_trainer_passes_epoch_to_fold_metric_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = TrainerConfig(
        epochs=2,
        early_stopping_patience=10,
        label_norm_strategy="none",
        device="cpu",
    )
    logger = MagicMock()
    trainer = Trainer(cfg=cfg, logger=logger)
    model = _EchoModel()
    normalizer = LabelNormalizer(strategy="none")
    normalizer.fit(np.array([1.0, 2.0], dtype=np.float32))

    monkeypatch.setattr(
        trainer,
        "_train_one_epoch",
        lambda _model, _loader, _optimizer: (
            0.1,
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
            {},
        ),
    )
    monkeypatch.setattr(trainer, "_build_scheduler", lambda _optimizer: None)

    val_metrics = iter(
        [
            {"mae": 0.5, "rmse": 0.5, "r2": 0.1, "pearson_r": 0.1},
            {"mae": 0.6, "rmse": 0.6, "r2": 0.2, "pearson_r": 0.2},
        ]
    )
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda *_args, **_kwargs: next(val_metrics),
    )

    batch = _FakeBatch([1.0, 2.0])
    trainer.fit(
        model=model,
        train_loader=[batch],
        val_loader=[batch],
        inverse_transform=normalizer.inverse_transform,
        fold_idx=0,
    )

    train_epochs = [
        call.kwargs["epoch"]
        for call in logger.log_fold_metrics.call_args_list
        if call.kwargs["split"] == "train"
    ]
    val_epochs = [
        call.kwargs["epoch"]
        for call in logger.log_fold_metrics.call_args_list
        if call.kwargs["split"] == "val"
    ]

    assert train_epochs == [0, 1]
    assert val_epochs == [0, 1]


def test_cross_validator_evaluates_test_split_with_best_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = TrainerConfig(
        n_folds=1,
        label_norm_strategy="none",
        checkpoint_dir=str(tmp_path),
        device="cpu",
    )
    cross_validator = CrossValidator(cfg=cfg)
    trainer = _FakeTrainer()
    created_models: list[_ConstantModel] = []

    def _model_factory() -> _ConstantModel:
        model = _ConstantModel()
        created_models.append(model)
        return model

    dataset = [
        torch_geometric.data.Data(
            x=torch.ones(1, 1),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            subject_id=f"subject_{idx}",
        )
        for idx in range(4)
    ]
    labels = np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(
        CrossValidator,
        "split",
        lambda self, dataset, labels: iter([([0, 1], [2], [3])]),
    )

    result = cross_validator.run(
        model_factory=_model_factory,
        dataset=dataset,
        labels=labels,
        trainer=trainer,
    )

    assert created_models[0].value.item() == pytest.approx(2.0)
    assert result.fold_test_metrics[0]["mae"] == pytest.approx(2.0)
    test_log_call = trainer.logger.log_fold_metrics.call_args_list[0]
    assert test_log_call.kwargs["split"] == "test"
    assert test_log_call.kwargs["epoch"] == 1