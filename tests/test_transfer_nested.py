import json
import pytest
import torch
from src.finetuning.transfer_nested import SourceBackboneProvider


def _make_source(tmp_path):
    root = tmp_path / "src_ckpt"
    (root / "rep_0" / "fold_0").mkdir(parents=True)
    torch.save({"backbone.weight": torch.ones(2, 2)},
               root / "rep_0" / "fold_0" / "model_best.pt")
    manifest = {"folds": [
        {"rep": 0, "fold": 0, "train_val_idx": [1, 2, 3], "test_idx": [0]},
    ]}
    (root / "fold_indices.json").write_text(json.dumps(manifest))
    return root


def test_assert_aligned_passes_on_match(tmp_path):
    root = _make_source(tmp_path)
    p = SourceBackboneProvider(root)
    p.assert_aligned(rep=0, fold=0, train_val_idx=[3, 1, 2], test_idx=[0])  # order-independent


def test_assert_aligned_raises_on_mismatch(tmp_path):
    root = _make_source(tmp_path)
    p = SourceBackboneProvider(root)
    with pytest.raises(ValueError, match="fold-index mismatch"):
        p.assert_aligned(rep=0, fold=0, train_val_idx=[1, 2], test_idx=[0, 3])


def test_state_dict_for_loads_weights(tmp_path):
    root = _make_source(tmp_path)
    p = SourceBackboneProvider(root)
    sd = p.state_dict_for(rep=0, fold=0)
    assert "backbone.weight" in sd


def test_missing_manifest_raises_filenotfound(tmp_path):
    # No fold_indices.json written -> fail closed at construction.
    with pytest.raises(FileNotFoundError):
        SourceBackboneProvider(tmp_path)


def test_empty_or_missing_folds_raises_valueerror(tmp_path):
    root = tmp_path / "empty_folds"
    root.mkdir()
    (root / "fold_indices.json").write_text(json.dumps({"folds": []}))
    with pytest.raises(ValueError, match="no 'folds' entries"):
        SourceBackboneProvider(root)

    root2 = tmp_path / "no_folds_key"
    root2.mkdir()
    (root2 / "fold_indices.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="no 'folds' entries"):
        SourceBackboneProvider(root2)


def test_invalid_variant_raises(tmp_path):
    root = _make_source(tmp_path)
    with pytest.raises(ValueError, match="variant must be 'best' or 'last'"):
        SourceBackboneProvider(root, variant="middle")


def test_assert_aligned_raises_on_unknown_fold(tmp_path):
    root = _make_source(tmp_path)
    p = SourceBackboneProvider(root)
    with pytest.raises(ValueError, match="fold-index mismatch"):
        p.assert_aligned(rep=9, fold=9, train_val_idx=[1, 2, 3], test_idx=[0])


def test_state_dict_for_missing_checkpoint_raises(tmp_path):
    # Manifest knows about (rep=0, fold=1) but no checkpoint file on disk.
    root = tmp_path / "missing_ckpt"
    root.mkdir()
    manifest = {"folds": [
        {"rep": 0, "fold": 1, "train_val_idx": [1, 2, 3], "test_idx": [0]},
    ]}
    (root / "fold_indices.json").write_text(json.dumps(manifest))
    p = SourceBackboneProvider(root)
    with pytest.raises(FileNotFoundError):
        p.state_dict_for(rep=0, fold=1)
