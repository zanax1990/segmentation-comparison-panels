# Segmentation Comparison Panel Generator

This utility creates publication-resolution comparison panels for microscopy segmentation experiments. Each panel places the input image, ground truth, multiple probabilistic predictions, the selected diffusion output, and a deterministic baseline in one row.

The current configuration covers LiveCell, TissueNet, and NIPS-style result folders.

## Expected layout

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

The prediction file may contain a 2D mask or a stack of samples. The script moves the inferred sample axis to the front and uses up to five samples in each panel.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Usage

Update `PRED_ROOT`, `FIG_DIR`, and the image IDs in `CONFIG`, then run:

```bash
python task1_multisample_panels.py
```

Figures are saved as 300 dpi PNG files using the pattern `<dataset>_fullpanel_<image_id>.png`.

## Limitations

Paths and experiment IDs are currently defined in the source file. Stack-axis detection is heuristic, and images with more than two spatial dimensions are reduced to one channel. The repository does not include example TIFF files, automated tests, or quantitative segmentation metrics.
