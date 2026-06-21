from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from learned_tta.selector_features import load_selector_features


@dataclass
class _FakeFeatureBundle:
    model: torch.nn.Module
    preprocess: object
    model_name: str = "fake_pretrained"
    pretrained: bool = True
    data_config: dict[str, object] | None = None


class _FakeFeatureModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        means = inputs.mean(dim=(1, 2, 3))
        return torch.stack([means, means + 1.0], dim=1)


def test_extract_selector_features_from_manifest_writes_cache(tmp_path: Path) -> None:
    from learned_tta.selector_feature_cache import extract_selector_features_from_manifest

    manifest_path = _write_manifest(tmp_path)
    output_path = tmp_path / "features.npz"
    bundle = _FakeFeatureBundle(
        model=_FakeFeatureModel(),
        preprocess=lambda image: torch.full((3, 4, 4), float(image.getpixel((0, 0))[0])),
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
