from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from learned_tta.selector_features import load_selector_features


class _FakeFeatureModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        means = inputs.mean(dim=(1, 2, 3))
        return torch.stack([means, means + 1.0], dim=1)


def test_extract_selector_features_from_manifest_writes_cache(tmp_path: Path) -> None:
    from learned_tta.selector_feature_cache import (
        FeatureExtractorBundle,
        extract_selector_features_from_manifest,
    )

    manifest_path = _write_manifest(tmp_path)
    output_path = tmp_path / "features.npz"
    bundle = FeatureExtractorBundle(
        model=_FakeFeatureModel(),
        preprocess=lambda image: torch.full((3, 4, 4), float(image.getpixel((0, 0))[0])),
        model_name="fake_pretrained",
        pretrained=True,
        data_config={"input_size": (3, 4, 4)},
    )

    written = extract_selector_features_from_manifest(
        manifest_path=manifest_path,
        output_path=output_path,
        bundle=bundle,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    loaded = load_selector_features(written)

    assert written == output_path
    assert loaded.split == "public_train"
    assert loaded.model_name == "fake_pretrained"
    assert loaded.image_ids == ["public_train-0", "public_train-1"]
    assert loaded.feature_names == ["feature_0000", "feature_0001"]
    assert loaded.features.shape == (2, 2)
    np.testing.assert_allclose(loaded.features[:, 1], loaded.features[:, 0] + 1.0)
    assert loaded.metadata["pretrained"] is True
    assert loaded.metadata["data_config"]["input_size"] == [3, 4, 4]


def test_build_timm_feature_extractor_uses_timm_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from learned_tta.selector_feature_cache import build_timm_feature_extractor

    created: dict[str, object] = {}
    timm_module = types.ModuleType("timm")
    timm_data_module = types.ModuleType("timm.data")

    def create_model(model_name: str, *, pretrained: bool, num_classes: int) -> torch.nn.Module:
        created["args"] = (model_name, pretrained, num_classes)
        return _FakeFeatureModel()

    def resolve_model_data_config(model: torch.nn.Module) -> dict[str, object]:
        created["model"] = model
        return {"input_size": (3, 4, 4)}

    def create_transform(**kwargs: object) -> object:
        return ("transform", kwargs)

    timm_module.__dict__["create_model"] = create_model
    timm_module.__dict__["data"] = timm_data_module
    timm_data_module.__dict__["resolve_model_data_config"] = resolve_model_data_config
    timm_data_module.__dict__["create_transform"] = create_transform
    monkeypatch.setitem(sys.modules, "timm", timm_module)
    monkeypatch.setitem(sys.modules, "timm.data", timm_data_module)

    bundle = build_timm_feature_extractor("fake_model", pretrained=False)

    assert created["args"] == ("fake_model", False, 0)
    assert bundle.model_name == "fake_model"
    assert bundle.pretrained is False
    assert bundle.data_config["input_size"] == (3, 4, 4)
    assert bundle.preprocess[0] == "transform"


def test_extract_selector_features_from_manifest_rejects_bad_manifests(tmp_path: Path) -> None:
    from learned_tta.selector_feature_cache import (
        FeatureExtractorBundle,
        extract_selector_features_from_manifest,
    )

    empty_manifest = tmp_path / "empty.csv"
    pd.DataFrame(columns=["split", "image_id", "class_idx", "class_name", "path"]).to_csv(
        empty_manifest,
        index=False,
    )
    bundle = FeatureExtractorBundle(
        model=_FakeFeatureModel(),
        preprocess=lambda image: torch.zeros((3, 4, 4)),
        model_name="fake_pretrained",
        pretrained=True,
        data_config={},
    )
    with pytest.raises(ValueError, match="at least one row"):
        extract_selector_features_from_manifest(
            manifest_path=empty_manifest,
            output_path=tmp_path / "empty_features.npz",
            bundle=bundle,
            batch_size=2,
            num_workers=0,
            device="cpu",
        )

    mixed_manifest = _write_manifest(tmp_path)
    rows = pd.read_csv(mixed_manifest)
    rows.loc[1, "split"] = "public_val"
    rows.to_csv(mixed_manifest, index=False)
    with pytest.raises(ValueError, match="single split"):
        extract_selector_features_from_manifest(
            manifest_path=mixed_manifest,
            output_path=tmp_path / "mixed_features.npz",
            bundle=bundle,
            batch_size=2,
            num_workers=0,
            device="cpu",
        )


def test_flatten_model_outputs_handles_shapes_and_errors() -> None:
    from learned_tta.selector_feature_cache import _flatten_model_outputs, _json_ready

    pooled = _flatten_model_outputs(torch.ones((2, 3, 4, 4)))
    flattened = _flatten_model_outputs([torch.ones((2, 3, 2))])

    assert pooled.shape == (2, 3)
    assert flattened.shape == (2, 6)
    assert _json_ready({"shape": (3, 4), "dtype": np.float32})["dtype"] == repr(np.float32)
    with pytest.raises(ValueError, match="empty"):
        _flatten_model_outputs([])
    with pytest.raises(ValueError, match="tensor"):
        _flatten_model_outputs({"bad": "output"})
    with pytest.raises(ValueError, match="shape"):
        _flatten_model_outputs(torch.ones(3))


def _write_manifest(root: Path) -> Path:
    rows = []
    for index in range(2):
        path = root / f"image_{index}.png"
        image = np.full((8, 8, 3), fill_value=index + 1, dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(path)
        rows.append(
            {
                "split": "public_train",
                "image_id": f"public_train-{index}",
                "class_idx": index,
                "class_name": f"class-{index}",
                "path": str(path),
            }
        )
    manifest_path = root / "public_train.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path
