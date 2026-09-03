from pathlib import Path

import numpy as np
import pytest
import tifffile

from segmentation_panels import build_panel, load_config, load_image_2d, load_prediction_stack


def write_panel_inputs(root: Path, shape: tuple[int, int] = (12, 10)) -> None:
    diffusion = root / "diffusion_results" / "demo"
    baseline = root / "baseline" / "demo"
    diffusion.mkdir(parents=True)
    baseline.mkdir(parents=True)
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    label = (image > image.mean()).astype(np.uint8)
    samples = np.stack([label, np.flipud(label), np.fliplr(label)])
    tifffile.imwrite(diffusion / "7.tif", image)
    tifffile.imwrite(diffusion / "7_label.tif", label)
    tifffile.imwrite(diffusion / "7_pred.tif", samples, photometric="minisblack")
    tifffile.imwrite(diffusion / "7_output.tif", samples[0])
    tifffile.imwrite(baseline / "7_output.tif", samples[1])


def test_load_image_2d(tmp_path):
    path = tmp_path / "image.tif"
    tifffile.imwrite(path, np.ones((8, 9), dtype=np.uint8))
    assert load_image_2d(path).shape == (8, 9)


def test_load_prediction_stack_with_explicit_axis(tmp_path):
    path = tmp_path / "pred.tif"
    tifffile.imwrite(path, np.ones((8, 9, 3), dtype=np.uint8))
    stack = load_prediction_stack(path, sample_axis=2)
    assert stack.shape == (3, 8, 9)
    assert np.isfinite(stack).all()


def test_build_panel_writes_png(tmp_path):
    prediction_root = tmp_path / "predictions"
    output_dir = tmp_path / "figures"
    write_panel_inputs(prediction_root)
    output = build_panel(prediction_root, output_dir, "demo", "7", max_samples=2)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_build_panel_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing required files"):
        build_panel(tmp_path, tmp_path / "figures", "demo", "7")


def test_load_config_accepts_nested_mapping(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"demo": {"image_ids": [1, "2"]}}', encoding="utf-8")
    assert load_config(path) == {"demo": ["1", "2"]}
