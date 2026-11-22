from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from docking_confidence.application.demo import make_demo
from docking_confidence.io.config import load_config, validate_config
from docking_confidence.io.dataset import load_paired_dataset


def test_dataset_rejects_infinite_features_and_missing_complex_ids(tmp_path: Path):
    source = tmp_path / "demo.csv"
    make_demo(str(source), n_complexes=5, runs_per_complex=2)
    config = load_config(Path(__file__).parents[1] / "configs" / "demo.yaml")
    features = config["features"]["columns"]

    frame = pd.read_csv(source)
    frame.loc[0, features[0]] = np.inf
    invalid = tmp_path / "invalid.csv"
    frame.to_csv(invalid, index=False)
    with pytest.raises(ValueError, match="finite"):
        load_paired_dataset(invalid, features)

    frame = pd.read_csv(source)
    frame.loc[0, "complex_id"] = None
    frame.to_csv(invalid, index=False)
    with pytest.raises(ValueError, match="missing identifiers"):
        load_paired_dataset(invalid, features)


def test_configuration_blocks_target_and_full_features():
    config = load_config(Path(__file__).parents[1] / "configs" / "demo.yaml")
    config["features"]["columns"].append("preserved")
    with pytest.raises(ValueError, match="unsupported|leaky"):
        validate_config(config)


def test_configuration_rejects_non_finite_numbers_and_unknown_graph_schema():
    config = load_config(Path(__file__).parents[1] / "configs" / "demo.yaml")
    config["model"]["C"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_config(config)

    config = load_config(Path(__file__).parents[1] / "configs" / "demo.yaml")
    config["contact_graph"]["schema_version"] = "future_graph_v99"
    with pytest.raises(ValueError, match="schema_version"):
        validate_config(config)
