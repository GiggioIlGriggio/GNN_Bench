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
