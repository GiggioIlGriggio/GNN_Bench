from pathlib import Path

from src.training.search_space import load_sweeper_params, parse_search_space

_FORBIDDEN = {"model.embedding_dim", "model.hidden_dim", "model.num_layers",
              "model.pooling", "model.jk_mode"}


def test_transfer_sweeper_has_no_backbone_arch_params():
    params = load_sweeper_params(Path("configs/sweeper/transfer_finetune.yaml"))
    specs = parse_search_space(params)
    names = {s.name for s in specs}
    assert not (names & _FORBIDDEN), f"backbone-arch params leak into transfer HPO: {names & _FORBIDDEN}"
    # must still tune optimiser + head
    assert "trainer.lr" in names


def test_source_age_pinned_sweeper_has_no_backbone_arch_params():
    params = load_sweeper_params(Path("configs/sweeper/source_age_pinned.yaml"))
    specs = parse_search_space(params)
    names = {s.name for s in specs}
    assert not (names & _FORBIDDEN), f"backbone-arch params leak into pinned-source HPO: {names & _FORBIDDEN}"
    # must still tune optimiser + head
    assert "trainer.lr" in names
