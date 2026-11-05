# Pinniped Dental Morphometrics

Custom Python tools for manually digitising anatomical landmarks and tracing two-dimensional tooth crown outlines from specimen photographs. The tools were developed for comparative studies of pinniped dental morphology and use the [napari](https://napari.org/) image viewer.

## Overview

This repository contains two complementary tools:

1. **Landmark digitiser** (`scripts/landmark_digitizer.py`): records predefined anatomical landmarks on a tooth image, including anterior and posterior crown-base points, cusp apices, and valleys between cusps. The landmark sequence is adapted to the selected cusp configuration (one to five cusps, including two-cusp positional alternatives). It saves labelled coordinates as CSV files and produces SVG quality-control plots.
2. **Crown-outline digitiser** (`scripts/crown_outline_digitizer.py`): lets a user manually trace the visible crown margin as a sequence of points, from the anterior crown base (BA) to the posterior crown base (BP). Previously saved landmarks are displayed as guides. The tool standardises the trace direction and saves the raw traced vertices as CSV files.

Both tools provide a graphical interface for reviewing work and moving through an image queue. Landmark placement and outline tracing are **manual**, not automated anatomical detection.

## Requirements

- Python 3.10 or newer (recommended; check compatibility with your installed napari version)
- `napari`, `numpy`, `pandas`, `magicgui`, `qtpy`, and `scikit-image`
- A working Qt graphical environment

Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install -r requirements.txt
```

An example `requirements.txt` is:

```text
napari[all]
numpy
pandas
magicgui
qtpy
scikit-image
```

For reproducibility, record the actual package versions used in the published analysis.

## Suggested repository structure

```text
pinniped-dental-morphometrics/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── landmark_digitizer.py
│   └── crown_outline_digitizer.py
├── metadata/
│   ├── image_queue.csv                  # local/private; do not commit
│   ├── landmarks_3cusp.json
│   ├── landmarks_4cusp.json
│   └── landmarks_5cusp.json
└── exports/                             # generated outputs; do not commit
```

**Important:** The current scripts calculate their base directory as the parent of the `scripts/` directory. Keep the scripts in `scripts/` unless you also change the `BASE` configuration in both files.

The three landmark JSON files are required by the first tool and should contain the landmark names in digitisation order. They must be included for the repository to be runnable. The scripts alone are not sufficient. The image queue is a CSV with a column named `image_path` containing paths to locally accessible images. Supported formats include JPG, PNG, TIFF and BMP.

## Workflow

### 1. Prepare the image queue

Create `metadata/image_queue.csv` locally:

```csv
image_path
/path/to/your/authorised-image-01.jpg
/path/to/your/authorised-image-02.jpg
```

These paths are placeholders. Do not publish your actual image queue if it exposes private directories or restricted specimen metadata.

### 2. Digitise anatomical landmarks

```bash
python scripts/landmark_digitizer.py
```

For each image, select the cusp configuration, click **Start Digitizing**, and place landmarks in the prompted sequence. Review or undo points as necessary, then save and proceed. The tool creates per-image landmark CSVs in `exports/<specimen-folder>/` and corresponding labelled SVG plots. Completed and skipped items are tracked so a session can be resumed.

### 3. Trace the crown outline

```bash
python scripts/crown_outline_digitizer.py
```

The second tool loads the existing landmark CSV for each image. Use the displayed anatomical points as guides and trace the visible enamel crown from BA to BP, then save. The saved outline contains manually placed path vertices, oriented consistently from BA to BP. **The current save routine does not automatically insert internal landmark coordinates into the outline CSV.**

### 4. Downstream analysis

The CSV outputs contain two-dimensional image coordinates (pixels). The scripts do **not** automatically convert pixel coordinates into physical units or calculate morphometric indices. Any scale calibration, measurement definitions, filtering and statistical analysis should be documented separately.

## Data handling and permissions

The repository should contain code and non-sensitive configuration templates only. Before making it public:

- Do not commit original museum or specimen photographs unless their reuse and redistribution are authorised.
- Do not commit private `image_queue.csv`, `skipped_files.csv`, or `exports/` data containing local paths, specimen identifiers or unpublished measurements.
- Inspect filenames, JSON files, comments and example data for private information or unpublished specimen metadata.
- Prefer synthetic or expressly licensed demonstration images and placeholder paths.
- Check all Git commits, not just the current folder: deleting a sensitive file from the latest version does not remove it from repository history.

Suggested `.gitignore`:

```gitignore
# Python / environment
.venv/
__pycache__/
*.py[cod]
.DS_Store

# Private local image queue and workflow state
metadata/image_queue.csv
metadata/skipped_files.csv

# Generated measurements and plots
exports/

# Local images and raw data (adapt to your directory structure)
images/
data/
```

**Note:** `.gitignore` prevents new untracked files from being added by normal Git operations; it does not remove files that have already been committed.

## Reproducibility and limitations

- Anatomical landmarks and crown traces are placed manually; results can depend on the operator and image quality.
- The landmark definitions are configuration-dependent and should be described in the associated publication.
- Coordinates are two-dimensional image coordinates, not automatically calibrated measurements.
- The software is a research workflow tool; it has not been presented here as a validated, general-purpose image segmentation package.
- The outline tool uses previously recorded landmarks as visual references, not as automatically integrated internal trace vertices.

## Citation

If this software is used in a publication, cite the archived version of this repository (for example, a release deposited on Zenodo with a DOI) and the napari software. Add your own repository DOI and preferred software citation once a release is published.

## Licence

**Choose and add a `LICENSE` file before public release.** A public GitHub repository does not automatically grant others permission to reuse or redistribute its code. Confirm that you have the rights to publish all included files.

## Contact

Open a GitHub issue to report a bug or request clarification once the repository is published.
