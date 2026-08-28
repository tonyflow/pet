from pathlib import Path

import pytest

from pet.config import load_data_config


def test_load_data_config() -> None:
    """Load the checked-in configuration into its typed representation."""
    config = load_data_config(Path("configs/data/oxford_pet.yaml"))
    assert config.transforms.image_size == (224, 224)
    assert config.dataset.split_seed == 1729


def test_rejects_unknown_schema(tmp_path: Path) -> None:
    """Reject configuration files using an unsupported schema version."""
    path = tmp_path / "config.yaml"
    path.write_text("schema_version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_data_config(path)
