# Pinniped Dental Morphometrics

A collection of custom Python tools for digitising, standardising, and analysing two-dimensional pinniped tooth morphology.

The workflow includes manual landmark placement, crown-outline tracing, coordinate standardisation, and automated extraction of dental morphological measurements.

The tools were developed for comparative studies of pinniped dental morphology and use the [napari](https://napari.org/) image viewer.

## Overview

This repository contains six Python scripts designed to support the analysis of pinniped postcanine tooth morphology.

The workflow consists of three main stages:

1. **Digitisation:** Manual identification of anatomical landmarks and tracing of tooth crown outlines from photographs.
2. **Coordinate standardisation:** Translation, rotation, and mirroring of landmark coordinates to ensure consistent positioning and orientation.
3. **Morphometric analysis:** Automated calculation of dental measurements, including cusp number, height, prominence, and arrangement.

The tools support teeth with varying numbers and arrangements of cusps, allowing morphological comparisons across different pinniped species.

## Software Requirements

The software was developed in Python and uses the following libraries:

- napari
- NumPy
- pandas
- magicgui
- qtpy
- scikit-image

Install the required dependencies using:

```bash
python -m pip install "napari[all]" numpy pandas magicgui qtpy scikit-image
```

Alternatively, if a `requirements.txt` file is provided:

```bash
python -m pip install -r requirements.txt
```

## Repository Structure

The recommended project structure is:

```text
pinniped-dental-morphometrics/
│
├── scripts/
│   ├── landmark_digitizer.py
│   ├── crown_outline_digitizer.py
│   ├── 01_translate.py
│   ├── 02_rotate.py
│   ├── 03_mirror.py
│   └── 04_measure_features.py
│
├── metadata/
│   ├── image_queue.csv
│   ├── landmarks_3cusp.json
│   ├── landmarks_4cusp.json
│   └── landmarks_5cusp.json
│
├── exports/
│
├── results/
│
├── requirements.txt
├── README.md
└── LICENSE
```

The `metadata/` directory contains image-processing instructions and landmark configurations. The `exports/` directory stores digitised coordinates and intermediate processing outputs, while `results/` contains the final morphometric measurements.

The image queue should contain paths to locally stored images. Users must create their own `image_queue.csv` before running the digitisation tools.

## 1. Landmark Digitisation

**Script:** `landmark_digitizer.py`

The first application allows users to manually identify and record anatomical landmarks on photographs of pinniped teeth.

The application uses napari to display each image and provides an interactive interface for placing landmarks in a predefined sequence.

Landmarks include:

- Anterior crown base (BA)
- Posterior crown base (BP)
- Primary cusp apex (A)
- Accessory cusp apices (B, C, D, E)
- Anterior and posterior crown valleys
- Valleys between adjacent cusps

The number and arrangement of landmarks depend on the number of cusps present in each tooth.

The software supports teeth with one to five cusps, including alternative configurations for two-cusped teeth.

### Features

- Manual placement of anatomical landmarks.
- Cusp-specific landmark configurations.
- Step-by-step guidance through landmark placement.
- Anatomical descriptions of individual landmarks.
- Undo functionality for correcting misplaced landmarks.
- Sequential processing of multiple images.
- Ability to skip unsuitable images.
- Automatic resumption of unfinished digitisation sessions.

### Outputs

For each tooth, the application generates:

- A CSV file containing landmark names and two-dimensional coordinates.
- A labelled SVG plot showing the landmarks and an approximate crown outline for quality control.

The coordinates are recorded in image-coordinate units.

### Running the application

```bash
python scripts/landmark_digitizer.py
```

Before running the application, ensure that the required landmark configuration files are present in `metadata/` and that `image_queue.csv` contains an `image_path` column with valid paths to the specimen images.

## 2. Crown-Outline Digitisation

**Script:** `crown_outline_digitizer.py`

The second application allows users to manually trace the visible enamel crown outline of each tooth.

The previously recorded anatomical landmarks are displayed as visual guides during tracing.

Users trace the crown from the anterior crown base (BA) to the posterior crown base (BP), following the visible crown surface.

### Features

- Manual tracing of the tooth crown outline.
- Display of previously digitised landmarks as anatomical guides.
- Visual distinction between cusp apices, valleys, and crown-base landmarks.
- Standardisation of tracing direction from anterior to posterior.
- Ability to clear and redraw crown outlines.
- Sequential processing of multiple images.
- Automatic resumption of unfinished sessions.

### Outputs

The application exports a CSV file containing the ordered two-dimensional coordinates of the traced crown outline.

The output also records the source image path, tracing direction, and digitisation timestamp.

The crown-outline coordinates are stored separately from the anatomical landmark coordinates.

### Running the application

```bash
python scripts/crown_outline_digitizer.py
```

The landmark digitisation stage must be completed first, as the crown-outline application requires the existing landmark files.

## 3. Coordinate Standardisation

Following landmark digitisation, three Python scripts standardise tooth position and orientation.

These scripts operate sequentially on the landmark coordinates and do not scale the teeth.

### 3.1 Coordinate Translation

**Script:** `01_translate.py`

This script translates all landmarks so that the midpoint between the anterior (BA) and posterior (BP) crown-base landmarks is positioned at the origin (0, 0).

Translation standardises tooth position without changing its shape, size, or orientation.

Outputs include:

- Translated landmark coordinates in CSV format.
- SVG plots showing the translated landmarks and crown baseline.

Run using:

```bash
python scripts/01_translate.py
```

### 3.2 Coordinate Rotation

**Script:** `02_rotate.py`

This script rotates the translated landmark coordinates so that:

- The line connecting BA and BP is horizontal.
- The primary cusp (A) is positioned above the baseline.

An additional 180-degree rotation is applied where necessary.

Rotation standardises tooth orientation without changing distances between landmarks.

Outputs include:

- Rotated landmark coordinates in CSV format.
- SVG plots for quality control.

Run using:

```bash
python scripts/02_rotate.py
```

### 3.3 Anatomical Orientation

**Script:** `03_mirror.py`

This script standardises the anterior–posterior orientation of each tooth.

Where necessary, the coordinates are reflected horizontally so that:

- Anterior is positioned on the left.
- Posterior is positioned on the right.

The script uses the anterior and posterior terminal valley landmarks (Vant and Vpost) to determine whether mirroring is required.

Outputs include:

- Standardised landmark coordinates in CSV format.
- SVG plots showing the final anatomical orientation.

Run using:

```bash
python scripts/03_mirror.py
```

## 4. Morphometric Feature Extraction

**Script:** `04_measure_features.py`

The final script automatically calculates morphological measurements from the standardised landmark coordinates.

These measurements characterise variation in cusp number, size, height, prominence, and arrangement.

### Measurements

**Cusp count**

The total number of recorded cusp apices (A–E).

**Crown baseline length**

The Euclidean distance between the anterior (BA) and posterior (BP) crown-base landmarks.

**Global cusp height**

The perpendicular distance between each cusp apex and the crown baseline (BA–BP).

**Accessory cusp vertical relief**

The vertical distance between each accessory cusp apex and its associated valley.

**Relative cusp height**

The height of each accessory cusp relative to the primary cusp (A).

**Relative primary cusp height**

The height of the primary cusp divided by crown baseline length.

**B–A–C angle**

The angle formed by the anterior accessory cusp (B), primary cusp (A), and posterior accessory cusp (C), with A as the vertex.

### Composite Morphological Metrics

Two custom metrics are also calculated to characterise accessory cusp prominence.

**Total Accessory Relief (TAR)**

TAR measures the combined vertical relief of accessory cusps relative to the height of the primary cusp.

\[
TAR = 1 + \frac{R_B+R_C+R_D+R_E}{H_A}
\]

Where:

- \(R_B\)–\(R_E\) are the vertical relief measurements of accessory cusps.
- \(H_A\) is the global height of the primary cusp.

**Total Cusp Height and Prominence (TCHP)**

TCHP combines accessory cusp heights and their local vertical relief relative to the primary cusp.

\[
TCHP = 1 + \frac{H_B+H_C+H_D+H_E+R_B+R_C+R_D+R_E}{H_A}
\]

Where:

- \(H_B\)–\(H_E\) are the global heights of accessory cusps.
- \(R_B\)–\(R_E\) are their local vertical relief measurements.
- \(H_A\) is the global height of the primary cusp.

For both metrics, absent accessory cusps contribute zero, and a tooth containing only the primary cusp has a value of 1.

These metrics are intended to capture aspects of dental morphology that are not represented by cusp count alone.

### Outputs

The script exports a CSV file containing the calculated measurements for each tooth:

```text
results/tooth_features.csv
```

Missing measurements are recorded as `NaN` where the required landmarks are unavailable.

The script also checks for duplicate image identifiers.

### Running the application

```bash
python scripts/04_measure_features.py
```

The script also supports recalculating measurements for an individual corrected tooth:

```bash
python scripts/04_measure_features.py --update path/to/corrected_tooth_3_mirrored.csv
```

This updates the relevant measurements in existing output datasets where matching records are found.

## 5. Complete Workflow

The recommended workflow is:

```text
Tooth photographs
       |
       v
Landmark digitisation
       |
       v
Crown-outline tracing
       |
       v
01_translate.py
       |
       v
02_rotate.py
       |
       v
03_mirror.py
       |
       v
04_measure_features.py
       |
       v
Morphometric measurements (CSV)
```

The crown-outline tracing application produces a separate coordinate dataset. The subsequent coordinate standardisation and feature extraction scripts operate on the landmark dataset.

Once landmark digitisation is complete, the processing scripts can be run sequentially:

```bash
python scripts/01_translate.py
python scripts/02_rotate.py
python scripts/03_mirror.py
python scripts/04_measure_features.py
```

## 6. Methodological Considerations

All measurements are derived from two-dimensional photographs of teeth in lateral view.

Consequently, the analysis captures variation in the visible crown profile, including cusp height, prominence, and arrangement, rather than the complete three-dimensional morphology of the tooth crown.

Landmarks are placed manually, and cusp identification depends on the anatomical criteria used during digitisation.

Coordinate translation, rotation, and mirroring standardise tooth position and orientation without scaling the original measurements.

Linear measurements therefore remain in image-coordinate units unless converted using external calibration information.

TAR and TCHP are custom morphological metrics and should not be interpreted as established measures of dental complexity without further validation.

## 7. Data Availability and Privacy

The repository contains the software used for dental landmark digitisation and morphometric analysis.

Specimen photographs, original museum images, unpublished specimen metadata, and research measurement datasets are not included unless explicitly provided with appropriate permissions.

Users should supply their own images and metadata when running the software.

The repository is intended to support reproducibility of the analytical workflow without distributing restricted museum material.

## 8. Citation

If you use these tools in published research, please cite the archived software release.

A version-specific DOI will be provided through Zenodo.

**Software citation:**

Author(s). (Year). *Pinniped Dental Morphometrics* (Version X.X.X) [Computer software]. Zenodo. DOI: [to be added].

The software uses napari, which should also be cited where appropriate.

## 9. Licence

See the `LICENSE` file for the terms governing use, modification, and redistribution of the software.

## 10. Acknowledgements

This software was developed for research on pinniped dental morphology and evolution.

It builds on open-source Python libraries, particularly napari, NumPy, pandas, magicgui, qtpy, and scikit-image.
