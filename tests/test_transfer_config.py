import pytest
from src.configs.transfer_config import TransferConfig


def test_disabled_by_default():
    assert TransferConfig().enabled is False


def test_enabled_requires_source_root():
    with pytest.raises(ValueError, match="source_checkpoint_root"):
        TransferConfig(enabled=True).validate_runtime()


def test_frozen_arm_roundtrip():
    c = TransferConfig(enabled=True, source_checkpoint_root="ck/age_id",
                       frozen_layers=["backbone"])
    c.validate_runtime()
    assert c.frozen_layers == ["backbone"]
