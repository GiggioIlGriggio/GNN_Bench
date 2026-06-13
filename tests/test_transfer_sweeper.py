# An exact-set (allowlist) assertion over the restricted sweeper configs guards
# against (a) any backbone-arch param leaking into a transfer/source HPO sweep --
# including ModelConfig fields not yet added, which a denylist could never catch --
# and (b) the two configs silently diverging from each other.
from pathlib import Path

import pytest

from src.training.search_space import load_sweeper_params, parse_search_space

_EXPECTED = {
    "trainer.lr",
    "trainer.weight_decay",
    "model.dropout",
    "model.head_hidden_dim",
    "model.head_num_layers",
}


@pytest.mark.parametrize(
    "sweeper",
    ["transfer_finetune.yaml", "source_age_pinned.yaml"],
)
def test_restricted_sweeper_tunes_exactly_allowed_params(sweeper):
    params = load_sweeper_params(Path("configs/sweeper") / sweeper)
    specs = parse_search_space(params)
    names = {s.name for s in specs}
    assert names == _EXPECTED, (
        f"unexpected: {names - _EXPECTED}, missing: {_EXPECTED - names}"
    )
