"""Create a small synthetic TIFF tree for an end-to-end panel check."""

import argparse
from pathlib import Path

import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("build/synthetic_predictions"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diffusion_dir = args.output_root / "diffusion_results" / "synthetic"
    baseline_dir = args.output_root / "baseline" / "synthetic"
    diffusion_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    y, x = np.mgrid[:64, :64]
    image = np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / 240.0).astype(np.float32)
    label = (((x - 32) ** 2 + (y - 32) ** 2) < 15**2).astype(np.uint8)
    samples = np.stack(
        [(((x - (30 + shift)) ** 2 + (y - 32) ** 2) < 15**2).astype(np.uint8) for shift in range(5)]
    )
    diffusion = samples[2]
    baseline = (((x - 31) ** 2 + (y - 31) ** 2) < 14**2).astype(np.uint8)

    tifffile.imwrite(diffusion_dir / "example.tif", image)
    tifffile.imwrite(diffusion_dir / "example_label.tif", label)
    tifffile.imwrite(diffusion_dir / "example_pred.tif", samples, photometric="minisblack")
    tifffile.imwrite(diffusion_dir / "example_output.tif", diffusion)
    tifffile.imwrite(baseline_dir / "example_output.tif", baseline)
    print(f"Wrote synthetic example to {args.output_root}")


if __name__ == "__main__":
    main()
