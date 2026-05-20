from __future__ import annotations

import nibabel as nib
import numpy as np

from brain_visualizaer.atlas import AtlasData, _infer_group, parcel_values_to_img
from brain_visualizaer.config import config_to_dict, load_config
from brain_visualizaer.io import fold_consensus_mask
from brain_visualizaer.node_metrics import (
    aggregate_edge_importance_on_nodes,
    build_group_importance_table,
    infer_grouping_name,
)


def test_fold_consensus_mask_keeps_edges_seen_in_enough_folds() -> None:
    fold_a = np.array(
        [
            [0.0, 9.0, 3.0],
            [9.0, 0.0, 1.0],
            [3.0, 1.0, 0.0],
        ]
    )
    fold_b = np.array(
        [
            [0.0, 8.0, 2.0],
            [8.0, 0.0, 5.0],
            [2.0, 5.0, 0.0],
        ]
    )
    fold_c = np.array(
        [
            [0.0, 1.0, 7.0],
            [1.0, 0.0, 6.0],
            [7.0, 6.0, 0.0],
        ]
    )

    mask = fold_consensus_mask(
        fold_matrices=np.stack([fold_a, fold_b, fold_c], axis=0),
        top_k=1,
        min_folds=2,
    )

    expected = np.array(
        [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ]
    )
    np.testing.assert_array_equal(mask, expected)


def test_load_config_round_trips_fold_consensus_settings(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "selection:",
                "  matrix: fold_mean",
                "  top_k_edges: 50",
                "  min_importance: 0.2",
                "  use_absolute: false",
                "  fold_consensus:",
                "    enabled: true",
                "    top_k_edges: 75",
                "    min_folds: 4",
            ]
        )
    )

    cfg = load_config(config_path)
    serialized = config_to_dict(cfg)

    assert cfg.selection.matrix == "fold_mean"
    assert cfg.selection.top_k_edges == 50
    assert cfg.selection.min_importance == 0.2
    assert cfg.selection.use_absolute is False
    assert cfg.selection.fold_consensus.enabled is True
    assert cfg.selection.fold_consensus.top_k_edges == 75
    assert cfg.selection.fold_consensus.min_folds == 4
    assert serialized["selection"]["fold_consensus"] == {
        "enabled": True,
        "top_k_edges": 75,
        "min_folds": 4,
    }


def test_load_config_round_trips_atlas_map_settings(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "plots:",
                "  atlas_map:",
                "    enabled: false",
                "    display_mode: ortho",
                "    cmap: viridis",
                "    black_bg: true",
                "    dim: 0.3",
            ]
        )
    )

    cfg = load_config(config_path)
    serialized = config_to_dict(cfg)

    assert cfg.plots.atlas_map.enabled is False
    assert cfg.plots.atlas_map.display_mode == "ortho"
    assert cfg.plots.atlas_map.cmap == "viridis"
    assert cfg.plots.atlas_map.black_bg is True
    assert cfg.plots.atlas_map.dim == 0.3
    assert serialized["plots"]["atlas_map"] == {
        "enabled": False,
        "display_mode": "ortho",
        "cmap": "viridis",
        "black_bg": True,
        "dim": 0.3,
    }


def test_load_config_uses_valid_default_atlas_map_display_mode(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("")

    cfg = load_config(config_path)

    assert cfg.plots.atlas_map.display_mode == "ortho"


def test_aggregate_edge_importance_on_nodes_sums_incident_edges() -> None:
    matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )

    values = aggregate_edge_importance_on_nodes(matrix)

    np.testing.assert_allclose(values, np.array([3.0, 4.0, 5.0]))


def test_build_group_importance_table_respects_yeo_order() -> None:
    atlas = AtlasData(
        name="toy",
        labels=["a", "b", "c"],
        coords=np.zeros((3, 3)),
        groups=["Default", "Vis", "Vis"],
        node_colors=["#000000", "#111111", "#111111"],
        sort_index=np.arange(3),
    )

    table = build_group_importance_table(np.array([1.0, 2.0, 3.0]), atlas)

    assert table["group"].tolist() == ["Vis", "Default"]
    assert table["total_node_importance"].tolist() == [5.0, 1.0]
    assert table["mean_node_importance"].tolist() == [2.5, 1.0]


def test_schaefer_control_alias_is_normalized_to_yeo7_control() -> None:
    assert _infer_group("7Networks_LH_Cont_Par_1") == "Control"

    atlas = AtlasData(
        name="toy",
        labels=[f"roi_{idx}" for idx in range(7)],
        coords=np.zeros((7, 3)),
        groups=[
            "Vis",
            "SomMot",
            "DorsAttn",
            "SalVentAttn",
            "Limbic",
            "Control",
            "Default",
        ],
        node_colors=["#000000"] * 7,
        sort_index=np.arange(7),
    )

    assert infer_grouping_name(atlas) == ("yeo7", "Yeo-7 network")


def test_parcel_values_to_img_maps_values_to_region_ids(tmp_path) -> None:
    atlas_path = tmp_path / "toy_atlas.nii.gz"
    atlas_img = nib.Nifti1Image(
        np.array(
            [
                [[0, 1], [2, 2]],
            ],
            dtype=np.int16,
        ),
        np.eye(4),
    )
    nib.save(atlas_img, atlas_path)
    atlas = AtlasData(
        name="toy",
        labels=["ROI_1", "ROI_2"],
        coords=np.zeros((2, 3)),
        groups=["Vis", "Default"],
        node_colors=["#000000", "#111111"],
        sort_index=np.arange(2),
        maps_path=atlas_path,
        region_ids=np.array([1, 2]),
    )

    value_img = parcel_values_to_img(atlas, np.array([0.5, 1.5]))
    value_data = np.asanyarray(value_img.dataobj)

    assert value_data[0, 0, 0] == 0.0
    assert value_data[0, 0, 1] == 0.5
    assert value_data[0, 1, 0] == 1.5
    assert value_data[0, 1, 1] == 1.5