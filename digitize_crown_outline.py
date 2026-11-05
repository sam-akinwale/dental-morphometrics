from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import napari
import numpy as np
import pandas as pd
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.io import imread


# ============================================================
# PATHS / SETTINGS
# ============================================================

BASE = Path(__file__).resolve().parent.parent

METADATA_DIR = BASE / "metadata"
EXPORTS_DIR = BASE / "exports"

QUEUE_FILE = METADATA_DIR / "image_queue.csv"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}

# Resume at the first image that does not already have
# a crown-outline CSV.
SKIP_COMPLETED_IMAGES = True

# Minimum number of vertices required for a crown trace.
MIN_TRACE_POINTS = 3

# ============================================================
# EXISTING CROWN LANDMARK ORDER
# ============================================================

CROWN_LANDMARK_ORDER = [
    "BA",
    "Vant",
    "B",
    "V(B-A)",
    "A",
    "V(A-C)",
    "C",
    "V(C-D)",
    "D",
    "V(D-E)",
    "E",
    "Vpost",
    "BP",
]

# ============================================================
# SESSION
# ============================================================

@dataclass
class OutlineSession:
    image_paths: list[Path]
    image_index: int = 0
    trace_active: bool = False
    out_path: Optional[Path] = None

    @property
    def image_path(self) -> Path:
        return self.image_paths[self.image_index]

    @property
    def total_images(self) -> int:
        return len(self.image_paths)

    @property
    def image_progress(self) -> str:
        return (
            f"Image {self.image_index + 1} "
            f"of {self.total_images}"
        )


# ============================================================
# IMAGE QUEUE
# ============================================================

def load_image_queue() -> list[Path]:

    if not QUEUE_FILE.exists():
        raise FileNotFoundError(
            f"Queue file not found: {QUEUE_FILE}"
        )

    queue_df = pd.read_csv(
        QUEUE_FILE,
        dtype=str,
    )

    if "image_path" not in queue_df.columns:
        raise ValueError(
            f"{QUEUE_FILE} must contain "
            f"an image_path column."
        )

    paths: list[Path] = []
    seen: set[Path] = set()

    for value in queue_df["image_path"].dropna():

        raw = str(value).strip()

        if not raw:
            continue

        path = Path(raw).expanduser()

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"Warning: unsupported image: {path}"
            )
            continue

        if not path.is_file():
            print(
                f"Warning: missing image: {path}"
            )
            continue

        if path in seen:
            continue

        seen.add(path)
        paths.append(path)

    if not paths:
        raise ValueError(
            "No usable images found in the queue."
        )

    return paths


# ============================================================
# OUTPUT PATHS
# ============================================================

def slugify(text: str) -> str:

    text = text.strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_") or "specimen"


def specimen_output_dir(
    image_path: Path,
) -> Path:

    specimen_slug = slugify(
        image_path.parent.name
    )

    return EXPORTS_DIR / specimen_slug


def canonical_outline_path(
    image_path: Path,
) -> Path:

    return (
        specimen_output_dir(image_path)
        / f"{image_path.stem}_crown_outline.csv"
    )


def has_saved_outline(
    image_path: Path,
) -> bool:

    out_dir = specimen_output_dir(
        image_path
    )

    if not out_dir.exists():
        return False

    canonical = canonical_outline_path(
        image_path
    )

    if canonical.exists():
        return True

    return any(
        out_dir.glob(
            f"{image_path.stem}"
            f"_crown_outline_v*.csv"
        )
    )


def find_outline_output_path(
    image_path: Path,
) -> Path:

    out_dir = specimen_output_dir(
        image_path
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate = canonical_outline_path(
        image_path
    )

    if not candidate.exists():
        return candidate

    version = 2

    while True:

        alternative = (
            out_dir
            / (
                f"{image_path.stem}"
                f"_crown_outline_v{version}.csv"
            )
        )

        if not alternative.exists():
            return alternative

        version += 1


# ============================================================
# EXISTING LANDMARK FILE
# ============================================================

def find_landmark_file(
    image_path: Path,
) -> Optional[Path]:

    out_dir = specimen_output_dir(
        image_path
    )

    canonical = (
        out_dir
        / f"{image_path.stem}_landmarks.csv"
    )

    if canonical.exists():
        return canonical

    candidates = sorted(
        out_dir.glob(
            f"{image_path.stem}_landmarks_v*.csv"
        )
    )

    if candidates:
        return candidates[-1]

    return None


def load_ba_bp(
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray]:

    landmark_file = find_landmark_file(
        image_path
    )

    if landmark_file is None:
        raise FileNotFoundError(
            "Could not find the existing "
            "landmark CSV for this tooth."
        )

    df = pd.read_csv(
        landmark_file
    )

    required = {
        "landmark",
        "x",
        "y",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"{landmark_file} does not contain "
            f"landmark, x, and y columns."
        )

    lookup = df.set_index(
        "landmark"
    )

    for name in [
        "BA",
        "BP",
    ]:

        if name not in lookup.index:
            raise ValueError(
                f"{name} not found in "
                f"{landmark_file}"
            )

    # Napari coordinates are (y, x).
    ba = np.array(
        [
            float(
                lookup.loc[
                    "BA",
                    "y",
                ]
            ),
            float(
                lookup.loc[
                    "BA",
                    "x",
                ]
            ),
        ],
        dtype=float,
    )

    bp = np.array(
        [
            float(
                lookup.loc[
                    "BP",
                    "y",
                ]
            ),
            float(
                lookup.loc[
                    "BP",
                    "x",
                ]
            ),
        ],
        dtype=float,
    )

    return ba, bp



def load_existing_landmarks(
    image_path: Path,
) -> pd.DataFrame:

    landmark_file = find_landmark_file(
        image_path
    )

    if landmark_file is None:
        raise FileNotFoundError(
            "Could not find the existing "
            "landmark CSV for this tooth."
        )

    df = pd.read_csv(
        landmark_file
    )

    required = {
        "landmark",
        "x",
        "y",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"{landmark_file} does not contain "
            f"landmark, x, and y columns."
        )

    return df

def integrate_landmarks_into_trace(
    coordinates: np.ndarray,
    landmark_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Integrate exact pre-existing anatomical landmarks into a
    dense BA->BP freehand crown trace.

    Each landmark is assigned to the nearest unused/forward
    location along the freehand trace, while preserving the
    known biological landmark order.

    The corresponding freehand vertex is replaced by the exact
    landmark coordinate, so landmarks do not create extra
    side-trips in the outline.
    """

    coordinates = np.asarray(
        coordinates,
        dtype=float,
    )

    # --------------------------------------------------------
    # Available landmarks in biological order
    # --------------------------------------------------------

    landmark_lookup = (
        landmark_df
        .drop_duplicates(
            subset="landmark",
            keep="first",
        )
        .set_index(
            "landmark"
        )
    )

    ordered_landmarks = [
        name
        for name in CROWN_LANDMARK_ORDER
        if name in landmark_lookup.index
    ]

    if (
        "BA" not in ordered_landmarks
        or "BP" not in ordered_landmarks
    ):
        raise ValueError(
            "BA and BP must both be present "
            "in the landmark file."
        )

    # --------------------------------------------------------
    # Force exact BA / BP first
    # --------------------------------------------------------

    ba = np.array(
        [
            float(
                landmark_lookup.loc[
                    "BA",
                    "y",
                ]
            ),
            float(
                landmark_lookup.loc[
                    "BA",
                    "x",
                ]
            ),
        ],
        dtype=float,
    )

    bp = np.array(
        [
            float(
                landmark_lookup.loc[
                    "BP",
                    "y",
                ]
            ),
            float(
                landmark_lookup.loc[
                    "BP",
                    "x",
                ]
            ),
        ],
        dtype=float,
    )

    # Orient manual trace BA -> BP.
    forward_error = (
        np.linalg.norm(
            coordinates[0] - ba
        )
        +
        np.linalg.norm(
            coordinates[-1] - bp
        )
    )

    reverse_error = (
        np.linalg.norm(
            coordinates[0] - bp
        )
        +
        np.linalg.norm(
            coordinates[-1] - ba
        )
    )

    trace_reversed = (
        reverse_error
        < forward_error
    )

    if trace_reversed:
        coordinates = (
            coordinates[::-1]
            .copy()
        )

    # Add exact endpoints.
    coordinates = np.vstack(
        [
            ba,
            coordinates,
            bp,
        ]
    )

    # Remove consecutive exact duplicates.
    differences = np.diff(
        coordinates,
        axis=0,
    )

    keep = np.concatenate(
        [
            [True],
            np.any(
                np.abs(
                    differences
                )
                > 1e-12,
                axis=1,
            ),
        ]
    )

    coordinates = coordinates[
        keep
    ]

    # --------------------------------------------------------
    # Start output as freehand
    # --------------------------------------------------------

    sources = np.full(
        len(coordinates),
        "freehand",
        dtype=object,
    )

    landmark_names = np.full(
        len(coordinates),
        "",
        dtype=object,
    )

    # --------------------------------------------------------
    # BA and BP are exact endpoints
    # --------------------------------------------------------

    coordinates[0] = ba
    sources[0] = "landmark"
    landmark_names[0] = "BA"

    coordinates[-1] = bp
    sources[-1] = "landmark"
    landmark_names[-1] = "BP"

    # --------------------------------------------------------
    # Snap internal landmarks to the trace in biological order
    # --------------------------------------------------------

    internal_landmarks = [
        name
        for name in ordered_landmarks
        if name not in {
            "BA",
            "BP",
        }
    ]

    previous_index = 0

    for landmark_number, name in enumerate(
        internal_landmarks
    ):

        landmark_coord = np.array(
            [
                float(
                    landmark_lookup.loc[
                        name,
                        "y",
                    ]
                ),
                float(
                    landmark_lookup.loc[
                        name,
                        "x",
                    ]
                ),
            ],
            dtype=float,
        )

        # Leave enough vertices for all remaining landmarks
        # plus the final BP.
        remaining_landmarks = (
            len(internal_landmarks)
            - landmark_number
            - 1
        )

        search_start = (
            previous_index + 1
        )

        search_stop = (
            len(coordinates)
            - 1
            - remaining_landmarks
        )

        if search_start >= search_stop:
            raise ValueError(
                "Not enough trace vertices to "
                "integrate all landmarks in order."
            )

        candidate_indices = np.arange(
            search_start,
            search_stop,
        )

        candidate_coords = coordinates[
            candidate_indices
        ]

        distances = np.linalg.norm(
            candidate_coords
            - landmark_coord,
            axis=1,
        )

        best_local = int(
            np.argmin(
                distances
            )
        )

        best_index = int(
            candidate_indices[
                best_local
            ]
        )

        # Replace nearest trace vertex with exact landmark.
        coordinates[
            best_index
        ] = landmark_coord

        sources[
            best_index
        ] = "landmark"

        landmark_names[
            best_index
        ] = name

        previous_index = best_index

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "point":
                np.arange(
                    len(coordinates)
                ),

            "y":
                coordinates[:, 0],

            "x":
                coordinates[:, 1],

            "point_source":
                sources,

            "landmark":
                landmark_names,
        }
    )

    return (
        result,
        trace_reversed,
    )

# ============================================================
# QUEUE NAVIGATION
# ============================================================

def next_pending_index(
    image_paths: list[Path],
    start: int,
) -> Optional[int]:

    for index in range(
        start,
        len(image_paths),
    ):

        image_path = image_paths[index]

        if (
            not SKIP_COMPLETED_IMAGES
            or not has_saved_outline(
                image_path
            )
        ):
            return index

    return None


# ============================================================
# TRACE EXTRACTION
# ============================================================

def get_trace_coordinates(
    shapes_layer,
) -> np.ndarray:

    if len(shapes_layer.data) == 0:
        raise ValueError(
            "No crown trace has been drawn."
        )

    if len(shapes_layer.data) > 1:
        raise ValueError(
            "More than one crown trace exists. "
            "Clear the layer and draw one path."
        )

    coordinates = np.asarray(
        shapes_layer.data[0],
        dtype=float,
    )

    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
    ):
        raise ValueError(
            "Unexpected crown-trace "
            "coordinate format."
        )

    if len(coordinates) < MIN_TRACE_POINTS:
        raise ValueError(
            f"Trace must contain at least "
            f"{MIN_TRACE_POINTS} points."
        )

    return coordinates


# ============================================================
# SAVE RAW TRACE
# ============================================================

def save_outline(
    session: OutlineSession,
    shapes_layer,
) -> None:

    coordinates = get_trace_coordinates(
        shapes_layer
    )

    # --------------------------------------------------------
    # Orient freehand trace BA -> BP
    # but DO NOT integrate internal landmarks
    # --------------------------------------------------------

    ba, bp = load_ba_bp(
        session.image_path
    )

    first = coordinates[0]
    last = coordinates[-1]


    forward_error = (
        np.linalg.norm(first - ba)
        +
        np.linalg.norm(last - bp)
    )

    reverse_error = (
        np.linalg.norm(first - bp)
        +
        np.linalg.norm(last - ba)
    )


    trace_reversed = (
        reverse_error
        < forward_error
    )


    if trace_reversed:

        coordinates = (
            coordinates[::-1]
            .copy()
        )

    # Remove consecutive duplicate points
    differences = np.diff(
        coordinates,
        axis=0,
    )

    keep = np.concatenate(
        [
            [True],
            np.any(
                np.abs(differences) > 1e-12,
                axis=1,
            ),
        ]
    )

    coordinates = coordinates[
        keep
    ]


    df = pd.DataFrame(
        {
            "point":
                np.arange(
                    len(coordinates)
                ),

            "y":
                coordinates[:, 0],

            "x":
                coordinates[:, 1],
        }
    )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    df["image_path"] = str(
        session.image_path
        .expanduser()
        .resolve()
    )

    df["trace_direction"] = (
        "BA_to_BP"
    )

    df[
        "trace_was_reversed_on_save"
    ] = trace_reversed


    df["digitized_at"] = (
        datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if session.out_path is None:

        session.out_path = (
            find_outline_output_path(
                session.image_path
            )
        )

    df.to_csv(
        session.out_path,
        index=False,
    )

    print(
        f"Saved {len(df)} crown-outline "
        f"points to {session.out_path}"
    )

# ============================================================
# CONTROL PANEL
# ============================================================

def build_control_widget(
    session: OutlineSession,
    viewer: napari.Viewer,
    image_layer,
    shapes_layer,
    endpoint_layer,
):

    root = QWidget()

    layout = QVBoxLayout(
        root
    )

    # --------------------------------------------------------
    # Heading
    # --------------------------------------------------------

    title = QLabel(
        "Pinniped Crown Outline Digitizer"
    )

    title.setStyleSheet(
        "font-size: 16px; "
        "font-weight: 600;"
    )

    layout.addWidget(
        title
    )

    queue_label = QLabel(
        session.image_progress
    )

    queue_label.setStyleSheet(
        "font-weight: 600;"
    )

    layout.addWidget(
        queue_label
    )

    image_label = QLabel()

    image_label.setWordWrap(
        True
    )

    layout.addWidget(
        image_label
    )

    status_label = QLabel(
        "Status: not yet traced"
    )

    status_label.setWordWrap(
        True
    )

    layout.addWidget(
        status_label
    )

    instruction = QLabel(
        "Click Start Crown Trace, then "
        "trace the visible enamel crown "
        "from BA to BP."
    )

    instruction.setWordWrap(
        True
    )

    layout.addWidget(
        instruction
    )

    endpoint_label = QLabel(
        "Existing landmarks are shown as guides.\n"
        "Red = cusp apex\n"
        "Blue = valley\n"
        "Light blue = BA\n"
        "Orange = BP\n\n"
        "Trace BA -> crown surface -> BP."
    )
    endpoint_label.setWordWrap(
        True
    )

    layout.addWidget(
        endpoint_label
    )

    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------

    start_button = QPushButton(
        "Start Crown Trace"
    )

    clear_button = QPushButton(
        "Undo / Clear Trace"
    )

    save_next_button = QPushButton(
        "Save & Next Tooth"
    )

    skip_button = QPushButton(
        "Skip This Tooth"
    )

    clear_button.setShortcut(
        QKeySequence(
            "Ctrl+Z"
        )
    )

    save_next_button.setEnabled(
        False
    )

    layout.addWidget(
        start_button
    )

    layout.addWidget(
        clear_button
    )

    layout.addWidget(
        save_next_button
    )

    layout.addWidget(
        skip_button
    )

    # --------------------------------------------------------
    # GUI helpers
    # --------------------------------------------------------

    def update_image_labels():

        queue_label.setText(
            session.image_progress
        )

        image_label.setText(
            f"Image: "
            f"{session.image_path.name}\n"
            f"Folder: "
            f"{session.image_path.parent.name}"
        )

    def update_save_button_text():

        upcoming = next_pending_index(
            session.image_paths,
            session.image_index + 1,
        )

        if upcoming is None:
            save_next_button.setText(
                "Save Final Tooth"
            )

        else:
            save_next_button.setText(
                "Save & Next Tooth"
            )

    # --------------------------------------------------------
    # BA / BP guide points
    # --------------------------------------------------------

    def load_endpoint_guides():

        landmark_df = (
            load_existing_landmarks(
                session.image_path
            )
        )

        coordinates = (
            landmark_df[
                [
                    "y",
                    "x",
                ]
            ]
            .to_numpy(
                dtype=float
            )
        )

        endpoint_layer.data = coordinates

        # --------------------------------------------------------
        # Landmark labels
        # --------------------------------------------------------

        endpoint_layer.properties = {
            "landmark":
                landmark_df[
                    "landmark"
                ]
                .astype(str)
                .tolist()
        }

        try:

            endpoint_layer.text = {
                "string": "{landmark}",
                "size": 9,
                "color": "white",
                "translation": [
                    6,
                    6,
                ],
            }

        except Exception as exc:

            print(
                "Could not display landmark "
                f"labels: {exc}"
            )

        # --------------------------------------------------------
        # Landmark colors
        # --------------------------------------------------------

        colors = []

        for name in landmark_df[
            "landmark"
        ]:

            name = str(
                name
            )

            if name == "BA":

                colors.append(
                    "deepskyblue"
                )

            elif name == "BP":

                colors.append(
                    "orange"
                )

            elif name.startswith(
                "V"
            ):

                colors.append(
                    "dodgerblue"
                )

            else:

                # A/B/C/D/E cusp apices
                colors.append(
                    "red"
                )

        try:

            endpoint_layer.face_color = (
                colors
            )

        except Exception as exc:

            print(
                "Could not assign individual "
                f"landmark colors: {exc}"
            )

        # Visual guide only.
        endpoint_layer.mode = (
            "pan_zoom"
        )

    def reset_current():

        session.trace_active = False
        session.out_path = None

        shapes_layer.data = []

        shapes_layer.mode = (
            "pan_zoom"
        )

        endpoint_layer.mode = (
            "pan_zoom"
        )

        save_next_button.setEnabled(
            False
        )

        status_label.setText(
            "Status: not yet traced"
        )

        instruction.setText(
            "Click Start Crown Trace, "
            "then trace the visible enamel "
            "crown from BA to BP."
        )

        update_image_labels()
        update_save_button_text()

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    def load_current_image():

        image = imread(
            str(
                session.image_path
            )
        )

        image_layer.data = image

        image_layer.name = (
            f"tooth: "
            f"{session.image_path.name}"
        )

        try:
            image_layer.reset_contrast_limits()

        except Exception:
            pass

        reset_current()

        load_endpoint_guides()

        try:
            viewer.reset_view()

        except Exception:
            pass

        print(
            f"Loaded "
            f"{session.image_progress}: "
            f"{session.image_path}"
        )

    # --------------------------------------------------------
    # Start tracing
    # --------------------------------------------------------

    def start_trace():

        # Remove any old trace.
        shapes_layer.data = []

        session.trace_active = True

        session.out_path = (
            find_outline_output_path(
                session.image_path
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Explicitly select the Shapes layer before entering
        # path-drawing mode. Otherwise another layer may remain
        # active and mouse clicks will not create path vertices.
        # ----------------------------------------------------

        viewer.layers.selection.active = (
            shapes_layer
        )

        shapes_layer.mode = (
            "add_path"
        )

        print(
            "Shapes layer mode:",
            shapes_layer.mode,
        )

        print(
            "Active layer:",
            viewer.layers.selection.active.name,
        )

        status_label.setText(
            "Status: tracing"
        )

        instruction.setText(
            "Click sequentially along the "
            "visible crown outline. "
            "Begin at BA and finish at BP. "
            "Double-click the final point "
            "to finish the path."
        )

    # --------------------------------------------------------
    # Detect trace
    # --------------------------------------------------------

    def on_shapes_change(
        event=None,
    ):

        if not session.trace_active:
            return

        if len(shapes_layer.data) != 1:
            save_next_button.setEnabled(
                False
            )
            return

        try:
            coordinates = (
                get_trace_coordinates(
                    shapes_layer
                )
            )

        except Exception:
            save_next_button.setEnabled(
                False
            )
            return

        if (
            len(coordinates)
            >= MIN_TRACE_POINTS
        ):

            save_next_button.setEnabled(
                True
            )

            status_label.setText(
                "Status: trace ready "
                "to review/save"
            )

    # --------------------------------------------------------
    # Clear trace
    # --------------------------------------------------------

    def clear_trace():

        shapes_layer.data = []

        session.trace_active = True

        viewer.layers.selection.active = (
            shapes_layer
        )

        shapes_layer.mode = (
            "add_path"
        )

        save_next_button.setEnabled(
            False
        )

        status_label.setText(
            "Status: tracing"
        )

        instruction.setText(
            "Trace cleared. Begin again "
            "at BA and trace the crown "
            "surface to BP."
        )

    # --------------------------------------------------------
    # Advance to next image
    # --------------------------------------------------------

    def advance_to_next_image(
        message: str,
    ):

        upcoming = next_pending_index(
            session.image_paths,
            session.image_index + 1,
        )

        if upcoming is None:

            session.trace_active = False

            shapes_layer.mode = (
                "pan_zoom"
            )

            endpoint_layer.mode = (
                "pan_zoom"
            )

            save_next_button.setEnabled(
                False
            )

            skip_button.setEnabled(
                False
            )

            start_button.setEnabled(
                False
            )

            instruction.setText(
                f"{message} "
                "The image queue is complete."
            )

            print(
                "Finished all crown outlines."
            )

            return

        session.image_index = upcoming

        try:
            load_current_image()

        except Exception as exc:

            session.trace_active = False

            save_next_button.setEnabled(
                False
            )

            instruction.setText(
                f"{message} "
                f"Could not load next image: "
                f"{exc}"
            )

            print(
                f"Could not load "
                f"{session.image_path}: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    def save_and_advance():

        try:
            save_outline(
                session,
                shapes_layer,
            )

        except Exception as exc:

            instruction.setText(
                f"Could not save trace: "
                f"{exc}"
            )

            print(
                f"Could not save trace: "
                f"{exc}"
            )

            return

        if session.out_path is not None:

            status_label.setText(
                f"Status: saved as "
                f"{session.out_path.name}"
            )

        advance_to_next_image(
            "Saved successfully."
        )

    # --------------------------------------------------------
    # Skip
    # --------------------------------------------------------

    def skip_current():

        if len(shapes_layer.data) > 0:

            response = QMessageBox.question(
                root,
                "Skip this tooth?",
                (
                    "This will discard the "
                    "current unsaved crown "
                    "trace and move to the "
                    "next tooth."
                ),
                (
                    QMessageBox.StandardButton.Yes
                    |
                    QMessageBox.StandardButton.No
                ),
                QMessageBox.StandardButton.No,
            )

            if (
                response
                != QMessageBox.StandardButton.Yes
            ):
                return

        advance_to_next_image(
            "Skipped."
        )

    # --------------------------------------------------------
    # Connections
    # --------------------------------------------------------

    start_button.clicked.connect(
        start_trace
    )

    clear_button.clicked.connect(
        clear_trace
    )

    save_next_button.clicked.connect(
        save_and_advance
    )

    skip_button.clicked.connect(
        skip_current
    )

    shapes_layer.events.data.connect(
        on_shapes_change
    )

    update_image_labels()
    update_save_button_text()

    return (
        root,
        load_current_image,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:
        image_paths = load_image_queue()

    except Exception as exc:

        print(
            f"Could not load image queue: "
            f"{exc}"
        )

        return

    start_index = next_pending_index(
        image_paths,
        0,
    )

# TEMPORARY QC:
# Jump directly to first Lobodon image
    #start_index = 46

    if start_index is None:

        print(
            "Every image already has "
            "a crown outline."
        )

        return

    session = OutlineSession(
        image_paths=image_paths,
        image_index=start_index,
    )

    try:
        initial_image = imread(
            str(
                session.image_path
            )
        )

    except Exception as exc:

        print(
            f"Could not open first image: "
            f"{exc}"
        )

        return

    viewer = napari.Viewer()

    # --------------------------------------------------------
    # Image layer
    # --------------------------------------------------------

    image_layer = viewer.add_image(
        initial_image,
        name=(
            f"tooth: "
            f"{session.image_path.name}"
        ),
    )

    # --------------------------------------------------------
    # Crown trace layer
    # --------------------------------------------------------

    shapes_layer = viewer.add_shapes(
        name="crown outline",
        shape_type="path",
        edge_color="lime",
        edge_width=3,
        face_color="transparent",
    )

    # --------------------------------------------------------
    # BA / BP guide layer
    #
    # This uses border_color / border_width because that is
    # the API used by your installed Napari version.
    # --------------------------------------------------------

    endpoint_layer = viewer.add_points(
        name="Existing landmarks",
        size=3,
        face_color="white",
        border_color="white",
        border_width=0,
    )

    endpoint_layer.mode = (
        "pan_zoom"
    )

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    (
        widget,
        load_current_image,
    ) = build_control_widget(
        session=session,
        viewer=viewer,
        image_layer=image_layer,
        shapes_layer=shapes_layer,
        endpoint_layer=endpoint_layer,
    )

    viewer.window.add_dock_widget(
        widget,
        area="right",
    )

    # --------------------------------------------------------
    # Initialize first image
    # --------------------------------------------------------

    try:
        load_current_image()

    except Exception as exc:

        print(
            f"Could not initialize "
            f"first image: {exc}"
        )

        return

    saved_count = sum(
        has_saved_outline(
            path
        )
        for path in image_paths
    )

    print(
        f"Loaded "
        f"{len(image_paths)} images."
    )

    print(
        f"Already outlined: "
        f"{saved_count}"
    )

    print(
        f"Starting at "
        f"{session.image_progress}: "
        f"{session.image_path}"
    )

    print(
        "Trace each crown from BA to BP."
    )

    napari.run()

    remaining = sum(
        not has_saved_outline(
            path
        )
        for path in image_paths
    )

    print(
        f"Session closed. "
        f"Images still pending: "
        f"{remaining}"
    )


if __name__ == "__main__":
    main()