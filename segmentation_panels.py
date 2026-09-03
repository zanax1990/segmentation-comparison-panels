"""Build comparison panels for microscopy segmentation outputs."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_image_2d(path: Path) -> np.ndarray:
    """Load a TIFF image and return one two-dimensional plane."""
    array = np.squeeze(tifffile.imread(path))
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] <= 4:
        return array[..., 0]
    raise ValueError(f"Expected a 2D image or channel-last image, found {array.shape} in {path}")


def load_prediction_stack(
    path: Path,
    *,
    sample_axis: int | None = None,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Load a TIFF prediction as an array shaped ``(samples, height, width)``.

    For three-dimensional data, the smallest dimension is used as the sample
    axis unless ``sample_axis`` is provided. Four-dimensional data require one
    channel to be selected; a dimension of size four or less is inferred when
    ``channel_axis`` is omitted.
    """
    array = np.squeeze(tifffile.imread(path))
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim not in (3, 4):
        raise ValueError(f"Expected a 2D, 3D, or 4D prediction, found {array.shape} in {path}")

    selected_sample_axis = int(np.argmin(array.shape)) if sample_axis is None else sample_axis
    selected_sample_axis %= array.ndim
    array = np.moveaxis(array, selected_sample_axis, 0)

    if array.ndim == 4:
        if channel_axis is not None:
            original_axes = list(range(4))
            moved_axes = [original_axes.pop(selected_sample_axis)] + original_axes
            selected_channel_axis = moved_axes.index(channel_axis % 4)
        else:
            candidates = [axis for axis in range(1, 4) if array.shape[axis] <= 4]
            if not candidates:
                raise ValueError(
                    f"Could not infer a channel axis in prediction shape {array.shape}; "
                    "pass --channel-axis"
                )
            selected_channel_axis = candidates[0]
        array = np.take(array, indices=0, axis=selected_channel_axis)

    if array.ndim != 3:
        raise ValueError(f"Prediction could not be reduced to (samples, height, width): {array.shape}")
    return array


def expected_paths(prediction_root: Path, dataset: str, image_id: str) -> dict[str, Path]:
    diffusion_dir = prediction_root / "diffusion_results" / dataset
    baseline_dir = prediction_root / "baseline" / dataset
    return {
        "input": diffusion_dir / f"{image_id}.tif",
        "label": diffusion_dir / f"{image_id}_label.tif",
        "samples": diffusion_dir / f"{image_id}_pred.tif",
        "diffusion": diffusion_dir / f"{image_id}_output.tif",
        "baseline": baseline_dir / f"{image_id}_output.tif",
    }


def build_panel(
    prediction_root: Path,
    output_dir: Path,
    dataset: str,
    image_id: str,
    *,
    max_samples: int = 5,
    sample_axis: int | None = None,
    channel_axis: int | None = None,
    dpi: int = 300,
) -> Path:
    """Build and save one comparison panel."""
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    paths = expected_paths(prediction_root, dataset, image_id)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required files:\n{missing_text}")

    image = load_image_2d(paths["input"])
    ground_truth = load_image_2d(paths["label"])
    predictions = load_prediction_stack(
        paths["samples"], sample_axis=sample_axis, channel_axis=channel_axis
    )
    diffusion = load_image_2d(paths["diffusion"])
    baseline = load_image_2d(paths["baseline"])

    spatial_shapes = {
        image.shape,
        ground_truth.shape,
        predictions.shape[1:],
        diffusion.shape,
        baseline.shape,
    }
    if len(spatial_shapes) != 1:
        raise ValueError(f"All panel images must share one spatial shape, found {spatial_shapes}")

    sample_count = min(max_samples, predictions.shape[0])
    panel_items: list[tuple[str, np.ndarray, str]] = [
        ("Input", image, "gray"),
        ("Ground truth", ground_truth, "nipy_spectral"),
    ]
    panel_items.extend(
        (f"Sample {index + 1}", predictions[index], "nipy_spectral")
        for index in range(sample_count)
    )
    panel_items.extend(
        [
            ("Diffusion output", diffusion, "nipy_spectral"),
            ("Baseline output", baseline, "nipy_spectral"),
        ]
    )

    figure, axes = plt.subplots(1, len(panel_items), figsize=(3 * len(panel_items), 3))
    for axis, (title, panel_image, color_map) in zip(np.atleast_1d(axes), panel_items):
        axis.imshow(panel_image, cmap=color_map)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"{dataset} — ID {image_id}", y=0.98)
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_fullpanel_{image_id}.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def load_config(path: Path) -> dict[str, list[str]]:
    """Load and validate a dataset-to-image-ID mapping."""
    with path.open(encoding="utf-8") as handle:
        raw_config = json.load(handle)
    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError("Configuration must be a non-empty JSON object")

    config: dict[str, list[str]] = {}
    for dataset, value in raw_config.items():
        image_ids = value.get("image_ids") if isinstance(value, dict) else value
        if not isinstance(dataset, str) or not isinstance(image_ids, list) or not image_ids:
            raise ValueError("Each dataset must map to a non-empty image_ids list")
        config[dataset] = [str(image_id) for image_id in image_ids]
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", type=Path, required=True, help="Root of the prediction tree")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG panels")
    parser.add_argument("--config", type=Path, required=True, help="JSON dataset and image-ID mapping")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--sample-axis", type=int)
    parser.add_argument("--channel-axis", type=int)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    created = 0
    for dataset, image_ids in config.items():
        for image_id in image_ids:
            try:
                output = build_panel(
                    args.pred_root,
                    args.output_dir,
                    dataset,
                    image_id,
                    max_samples=args.max_samples,
                    sample_axis=args.sample_axis,
                    channel_axis=args.channel_axis,
                    dpi=args.dpi,
                )
                print(f"Saved {output}")
                created += 1
            except FileNotFoundError as error:
                if not args.skip_missing:
                    raise
                print(f"Skipped {dataset}/{image_id}: {error}")
    print(f"Created {created} panel(s)")


if __name__ == "__main__":
    main()
