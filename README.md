# Segmentation Comparison Panels

This repository creates publication-resolution comparison panels for microscopy segmentation experiments. Each panel places the input image, ground truth, probabilistic prediction samples, a selected diffusion output, and a deterministic baseline in one row.

The utility supports LiveCell, TissueNet, and NIPS-style result folders, but the image IDs and paths are supplied through a JSON configuration and command-line arguments rather than source-code edits.

## Expected input layout

```text
predictions/
├── diffusion_results/
│   └── <dataset>/
│       ├── <image_id>.tif
│       ├── <image_id>_label.tif
│       ├── <image_id>_pred.tif
│       └── <image_id>_output.tif
└── baseline/
    └── <dataset>/
        └── <image_id>_output.tif
```

The prediction file may be a 2D mask or a stack. Three-dimensional stacks use the smallest dimension as the sample axis by default. For ambiguous arrays, pass `--sample-axis`; four-dimensional arrays can also use `--channel-axis`.

## Installation

```bash
git clone https://github.com/zanax1990/segmentation-comparison-panels.git
cd segmentation-comparison-panels
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Usage

```bash
python task1_multisample_panels.py \
  --pred-root /path/to/predictions \
  --output-dir figures \
  --config config/default_config.json \
  --max-samples 5
```

Figures are saved as 300 dpi PNG files named `<dataset>_fullpanel_<image_id>.png`. Use `--skip-missing` when a configuration intentionally contains IDs that are not available locally.

## Synthetic end-to-end check

The repository can generate a small synthetic TIFF tree and build a panel without external research data:

```bash
python examples/create_synthetic_example.py
python task1_multisample_panels.py \
  --pred-root build/synthetic_predictions \
  --output-dir build/figures \
  --config config/synthetic_config.json
```

This example checks file handling and panel layout only. It is not a segmentation result or benchmark.

## Tests

```bash
pytest
```

Tests cover 2D loading, explicit sample-axis handling, finite arrays, panel creation, configuration parsing, and missing-file errors. GitHub Actions also runs the synthetic end-to-end command.

## Limitations

- sample- and channel-axis inference is heuristic unless axes are supplied;
- the script selects one channel from channel-containing TIFFs;
- panels provide qualitative comparison and do not calculate segmentation metrics;
- research images and prediction outputs are not distributed in this repository.
