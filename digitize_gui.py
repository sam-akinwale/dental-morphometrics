from __future__ import annotations

import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import napari
import numpy as np
import pandas as pd
from magicgui import magicgui
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget
from skimage.io import imread


BASE = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE / "metadata"
EXPORTS_DIR = BASE / "exports"
QUEUE_FILE = METADATA_DIR / "image_queue.csv"
SKIPPED_FILE = METADATA_DIR / "skipped_files.csv"

LANDMARK_FILE_MAP = {
    "3": METADATA_DIR / "landmarks_3cusp.json",
    "4": METADATA_DIR / "landmarks_4cusp.json",
    "5": METADATA_DIR / "landmarks_5cusp.json",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# When True, the program begins at the first queue entry that does not already
# have a landmark CSV in exports/. This makes it easy to resume later.
SKIP_COMPLETED_IMAGES = True


LANDMARK_DESCRIPTIONS = {
    "BA": "Anterior-most point where the enamel crown meets the root.",
    "BP": "Posterior-most point where the enamel crown meets the root.",
    "Vant": "Terminal anterior valley at the anterior end of the crown.",
    "Vpost": "Terminal posterior valley at the posterior end of the crown.",
    "B": "Apex of cusp B (anterior).",
    "A": "Apex of cusp A (primary / tallest cusp).",
    "C": "Apex of cusp C (posterior).",
    "D": "Apex of cusp D.",
    "E": "Apex of cusp E.",
    "V(B-A)": "Deepest valley between cusp B (anterior) and cusp A.",
    "V(A-C)": "Deepest valley between cusp A and cusp C (posterior).",
    "V(C-D)": "Deepest valley between cusp C (posterior) and cusp D.",
    "V(D-E)": "Deepest valley between cusp D and cusp E.",
}


# Biological left-to-right outline used in the SVG quality-control plot.
CROWN_OUTLINE_ORDER = [
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
]


@dataclass
class DigitizeSession:
    image_paths: list[Path]
    image_index: int = 0
    landmark_names: list[str] = field(default_factory=list)
    clicked_points: list[tuple[float, float]] = field(default_factory=list)
    active: bool = False
    current_index: int = 0
    out_path: Optional[Path] = None

    @property
    def image_path(self) -> Path:
        return self.image_paths[self.image_index]

    @property
    def total_images(self) -> int:
        return len(self.image_paths)

    @property
    def image_progress(self) -> str:
        return f"Image {self.image_index + 1} of {self.total_images}"

    @property
    def total_landmarks(self) -> int:
        return len(self.landmark_names)

    @property
    def is_complete(self) -> bool:
        return (
            self.total_landmarks > 0
            and len(self.clicked_points) == self.total_landmarks
        )

    @property
    def current_landmark(self) -> str:
        if not self.landmark_names:
            return "Select cusp count to begin"
        if self.current_index >= len(self.landmark_names):
            return "Finished"
        return self.landmark_names[self.current_index]

    @property
    def previous_landmark(self) -> str:
        if not self.landmark_names or self.current_index <= 0:
            return "—"
        return self.landmark_names[self.current_index - 1]

    @property
    def next_landmark(self) -> str:
        if not self.landmark_names:
            return "—"
        nxt = self.current_index + 1
        if nxt >= len(self.landmark_names):
            return "—"
        return self.landmark_names[nxt]

    @property
    def landmark_progress(self) -> str:
        if not self.landmark_names:
            return "0 / 0"
        completed = min(self.current_index, len(self.landmark_names))
        return f"{completed} / {len(self.landmark_names)}"

    def clear_landmarks(self) -> None:
        self.landmark_names.clear()
        self.clicked_points.clear()
        self.active = False
        self.current_index = 0
        self.out_path = None


def load_image_queue() -> list[Path]:
    if not QUEUE_FILE.exists():
        raise FileNotFoundError(
            f"Queue file not found: {QUEUE_FILE}\n"
            "Create metadata/image_queue.csv with an image_path column."
        )

    queue_df = pd.read_csv(QUEUE_FILE, dtype=str)
    if "image_path" not in queue_df.columns:
        raise ValueError(
            f"{QUEUE_FILE} must contain a column named 'image_path'."
        )

    paths: list[Path] = []
    seen: set[Path] = set()
    missing: list[Path] = []
    unsupported: list[Path] = []

    for value in queue_df["image_path"].dropna():
        raw = str(value).strip()
        if not raw:
            continue

        path = Path(raw).expanduser()

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            unsupported.append(path)
            continue

        if not path.is_file():
            missing.append(path)
            continue

        if path in seen:
            continue

        seen.add(path)
        paths.append(path)

    if missing:
        print(f"Warning: skipped {len(missing)} missing queue file(s).")
        for path in missing[:10]:
            print(f"  Missing: {path}")
        if len(missing) > 10:
            print(f"  ...and {len(missing) - 10} more")

    if unsupported:
        print(f"Warning: skipped {len(unsupported)} unsupported file(s).")

    if not paths:
        raise ValueError(f"No usable image paths were found in {QUEUE_FILE}.")

    return paths


def load_landmark_names(cusp_count: str) -> list[str]:
    # 1-, 2-, and 2-pos-cusp teeth use the 3-cusp landmark file
    # as their starting point.
    if cusp_count in {"1", "2", "2-pos"}:
        landmark_file = LANDMARK_FILE_MAP["3"]
    else:
        landmark_file = LANDMARK_FILE_MAP[cusp_count]

    if not landmark_file.exists():
        raise FileNotFoundError(
            f"Missing landmark file: {landmark_file}. Create it in metadata/."
        )

    names = json.loads(landmark_file.read_text())

    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError(f"{landmark_file} must contain a JSON list of strings.")

    # For a 1-cusp tooth:
    # Do NOT record B, V(B-A), C, or V(A-C).
    if cusp_count == "1":
        excluded = {"B", "V(B-A)", "C", "V(A-C)"}
        names = [name for name in names if name not in excluded]

    # For a standard 2-cusp tooth:
    # Record B and A, with V(B-A).
    # Do NOT record C or V(A-C).
    elif cusp_count == "2":
        excluded = {"C", "V(A-C)"}
        names = [name for name in names if name not in excluded]

    # For a posterior 2-cusp tooth:
    # Record A and C, with V(A-C).
    # Do NOT record B or V(B-A).
    elif cusp_count == "2-pos":
        excluded = {"B", "V(B-A)"}
        names = [name for name in names if name not in excluded]

    return names


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "specimen"


def specimen_output_dir(image_path: Path) -> Path:
    specimen_slug = slugify(image_path.parent.name)
    return EXPORTS_DIR / specimen_slug


def canonical_output_path(image_path: Path) -> Path:
    return specimen_output_dir(image_path) / f"{image_path.stem}_landmarks.csv"


def has_saved_output(image_path: Path) -> bool:
    out_dir = specimen_output_dir(image_path)

    if not out_dir.exists():
        return False

    canonical = canonical_output_path(image_path)

    if canonical.exists():
        return True

    return any(out_dir.glob(f"{image_path.stem}_landmarks_v*.csv"))


def path_key(path: Path) -> str:
    """Return a consistent absolute path string for queue bookkeeping."""
    return str(path.expanduser().resolve())


def load_skipped_paths() -> set[str]:
    """Load image paths already recorded in metadata/skipped_files.csv."""
    if not SKIPPED_FILE.exists():
        return set()

    try:
        skipped_df = pd.read_csv(SKIPPED_FILE, dtype=str)
    except pd.errors.EmptyDataError:
        return set()

    if "image_path" not in skipped_df.columns:
        print(
            f"Warning: {SKIPPED_FILE} has no 'image_path' column; "
            "its entries cannot be used to resume the queue."
        )
        return set()

    return {
        path_key(Path(value))
        for value in skipped_df["image_path"].dropna()
        if str(value).strip()
    }


def is_skipped(image_path: Path) -> bool:
    return path_key(image_path) in load_skipped_paths()


def record_skipped_file(image_path: Path) -> None:
    """Record a skipped image without creating landmark CSV or SVG output."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    image_key = path_key(image_path)

    columns = ["file_name", "image_path", "parent_folder", "skipped_at"]

    if SKIPPED_FILE.exists():
        try:
            skipped_df = pd.read_csv(SKIPPED_FILE, dtype=str)
        except pd.errors.EmptyDataError:
            skipped_df = pd.DataFrame(columns=columns)
    else:
        skipped_df = pd.DataFrame(columns=columns)

    for column in columns:
        if column not in skipped_df.columns:
            skipped_df[column] = ""

    existing_paths = {
        path_key(Path(value))
        for value in skipped_df["image_path"].dropna()
        if str(value).strip()
    }

    if image_key in existing_paths:
        print(f"Already recorded as skipped: {image_path}")
        return

    new_row = pd.DataFrame(
        [
            {
                "file_name": image_path.name,
                "image_path": image_key,
                "parent_folder": image_path.parent.name,
                "skipped_at": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
            }
        ]
    )

    skipped_df = pd.concat(
        [skipped_df[columns], new_row],
        ignore_index=True,
    )

    skipped_df.to_csv(SKIPPED_FILE, index=False)

    print(f"Recorded skipped image in {SKIPPED_FILE}: {image_path}")


def find_output_path(image_path: Path) -> Path:
    out_dir = specimen_output_dir(image_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = canonical_output_path(image_path)

    if not candidate.exists():
        return candidate

    version = 2

    while True:
        alternative = out_dir / f"{image_path.stem}_landmarks_v{version}.csv"

        if not alternative.exists():
            return alternative

        version += 1


def next_pending_index(
    image_paths: list[Path],
    start: int,
) -> Optional[int]:
    """Find the next image that has been neither saved nor skipped."""
    skipped_paths = load_skipped_paths()

    for index in range(start, len(image_paths)):
        image_path = image_paths[index]
        saved = has_saved_output(image_path)
        skipped = path_key(image_path) in skipped_paths

        if not SKIP_COMPLETED_IMAGES or (not saved and not skipped):
            return index

    return None


def save_landmark_svg(df: pd.DataFrame, out_plot: Path) -> None:
    """Save a compact SVG showing the crown outline and labelled landmarks."""
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    min_x, max_x = float(x.min()), float(x.max())
    min_y, max_y = float(y.min()), float(y.max())

    data_width = max(max_x - min_x, 1.0)
    data_height = max(max_y - min_y, 1.0)

    canvas_width = 800
    canvas_height = 500
    margin = 55

    usable_width = canvas_width - (2 * margin)
    usable_height = canvas_height - (2 * margin)

    scale = min(
        usable_width / data_width,
        usable_height / data_height,
    )

    def transform(
        x_value: float,
        y_value: float,
    ) -> tuple[float, float]:
        # Napari/image coordinates use y increasing downward, so the SVG does too.
        svg_x = margin + (x_value - min_x) * scale
        svg_y = margin + (y_value - min_y) * scale

        return svg_x, svg_y

    coordinates = {
        str(landmark): (float(x), float(y))
        for landmark, x, y in df[
            ["landmark", "x", "y"]
        ].itertuples(index=False, name=None)
    }

    outline_names = [
        name
        for name in CROWN_OUTLINE_ORDER
        if name in coordinates
    ]

    outline_points = " ".join(
        (
            f"{transform(*coordinates[name])[0]:.2f},"
            f"{transform(*coordinates[name])[1]:.2f}"
        )
        for name in outline_names
    )

    svg_elements: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{canvas_width}" '
            f'height="{canvas_height}" '
            f'viewBox="0 0 {canvas_width} {canvas_height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<rect x="{margin}" y="{margin}" '
            f'width="{usable_width}" '
            f'height="{usable_height}" '
            'fill="none" stroke="#d0d0d0" stroke-width="1"/>'
        ),
        (
            f'<text x="{canvas_width / 2:.1f}" y="24" '
            'text-anchor="middle" '
            'font-family="Arial, sans-serif" '
            'font-size="16">'
            'Landmark QC plot</text>'
        ),
        (
            f'<text x="{canvas_width / 2:.1f}" '
            f'y="{canvas_height - 12}" '
            'text-anchor="middle" '
            'font-family="Arial, sans-serif" '
            'font-size="11">'
            'x coordinate</text>'
        ),
        (
            f'<text x="14" '
            f'y="{canvas_height / 2:.1f}" '
            'text-anchor="middle" '
            'font-family="Arial, sans-serif" '
            'font-size="11" '
            f'transform="rotate(-90 14 {canvas_height / 2:.1f})">'
            'y coordinate (increases downward)</text>'
        ),
    ]

    if len(outline_names) >= 2:
        svg_elements.append(
            f'<polyline points="{outline_points}" '
            'fill="none" '
            'stroke="#555555" '
            'stroke-width="2" '
            'stroke-linejoin="round" '
            'stroke-linecap="round"/>'
        )

    if "BA" in coordinates and "BP" in coordinates:
        ba_x, ba_y = transform(*coordinates["BA"])
        bp_x, bp_y = transform(*coordinates["BP"])

        svg_elements.append(
            f'<line x1="{ba_x:.2f}" '
            f'y1="{ba_y:.2f}" '
            f'x2="{bp_x:.2f}" '
            f'y2="{bp_y:.2f}" '
            'stroke="#111111" stroke-width="2"/>'
        )

    for landmark, x, y in df[
        ["landmark", "x", "y"]
    ].itertuples(index=False, name=None):
        name = str(landmark)

        point_x, point_y = transform(
            float(x),
            float(y),
        )

        if name in {"BA", "BP"}:
            fill = "#111111"
        elif name.startswith("V"):
            fill = "#1f77b4"
        else:
            fill = "#d62728"

        svg_elements.append(
            f'<circle cx="{point_x:.2f}" '
            f'cy="{point_y:.2f}" '
            'r="5" '
            f'fill="{fill}" '
            'stroke="#ffffff" '
            'stroke-width="1"/>'
        )

        svg_elements.append(
            f'<text x="{point_x + 7:.2f}" '
            f'y="{point_y - 7:.2f}" '
            'font-family="Arial, sans-serif" '
            'font-size="12">'
            f'{name}</text>'
        )

    svg_elements.append("</svg>")

    out_plot.write_text("\n".join(svg_elements))


def save_session(session: DigitizeSession) -> None:
    if not session.is_complete:
        raise ValueError(
            f"Expected {session.total_landmarks} points, "
            f"but got {len(session.clicked_points)}."
        )

    coordinates = np.asarray(
        session.clicked_points,
        dtype=float,
    )

    df = pd.DataFrame(
        coordinates,
        columns=["y", "x"],
    )

    df.insert(
        0,
        "landmark",
        session.landmark_names,
    )

    if session.out_path is None:
        session.out_path = find_output_path(
            session.image_path
        )

    df.to_csv(
        session.out_path,
        index=False,
    )

    print(
        f"Saved {len(df)} landmarks to "
        f"{session.out_path}"
    )

    try:
        out_plot = session.out_path.with_suffix(".svg")
        save_landmark_svg(df, out_plot)

        print(
            f"Saved landmark plot to "
            f"{out_plot}"
        )

    except Exception as exc:
        # A plotting error should never prevent the CSV from being saved.
        print(
            f"Could not save landmark plot: "
            f"{exc}"
        )


def describe_landmark(name: str) -> str:
    if name in LANDMARK_DESCRIPTIONS:
        return LANDMARK_DESCRIPTIONS[name]

    if name.isalpha() and len(name) == 1:
        if name == "A":
            return "Apex of cusp A (primary / tallest cusp)."

        return f"Apex of cusp {name}."

    return "No description available yet."


def build_control_widget(
    session: DigitizeSession,
    viewer: napari.Viewer,
    image_layer,
    points_layer,
):
    root = QWidget()
    layout = QVBoxLayout(root)

    title = QLabel("Pinniped Cusp Digitizer")
    title.setStyleSheet(
        "font-size: 16px; font-weight: 600;"
    )
    layout.addWidget(title)

    queue_label = QLabel(
        session.image_progress
    )
    queue_label.setStyleSheet(
        "font-weight: 600;"
    )
    layout.addWidget(queue_label)

    image_label = QLabel()
    image_label.setWordWrap(True)
    layout.addWidget(image_label)

    status_label = QLabel(
        "Status: not yet saved"
    )
    status_label.setWordWrap(True)
    layout.addWidget(status_label)

    instruction = QLabel(
        "Choose cusp count, then click Start Digitizing."
    )
    instruction.setWordWrap(True)
    layout.addWidget(instruction)

    current_label = QLabel(
        "Current landmark: —"
    )
    previous_label = QLabel(
        "Previous: —"
    )
    next_label = QLabel(
        "Next: —"
    )
    progress_label = QLabel(
        "Landmarks: 0 / 0"
    )
    description_label = QLabel(
        "Description: —"
    )
    description_label.setWordWrap(True)

    layout.addWidget(current_label)
    layout.addWidget(previous_label)
    layout.addWidget(next_label)
    layout.addWidget(progress_label)
    layout.addWidget(description_label)

    save_next_button = QPushButton(
        "Save & Next Tooth"
    )
    save_next_button.setEnabled(False)

    skip_button = QPushButton(
        "Skip This Tooth"
    )
    skip_button.setToolTip(
        "Record this image in metadata/skipped_files.csv "
        "and move to the next tooth."
    )

    def update_image_labels() -> None:
        queue_label.setText(
            session.image_progress
        )

        image_label.setText(
            f"Image: {session.image_path.name}\n"
            f"Folder: {session.image_path.parent.name}"
        )

    def update_landmark_labels() -> None:
        current = session.current_landmark

        current_label.setText(
            f"Current landmark: {current}"
        )

        previous_label.setText(
            f"Previous: {session.previous_landmark}"
        )

        next_label.setText(
            f"Next: {session.next_landmark}"
        )

        progress_label.setText(
            f"Landmarks: {session.landmark_progress}"
        )

        description_label.setText(
            f"Description: {describe_landmark(current)}"
        )

    def update_save_button_text() -> None:
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

    def reset_for_current_image() -> None:
        session.clear_landmarks()

        points_layer.data = []
        points_layer.mode = "pan_zoom"

        status_label.setText(
            "Status: not yet saved"
        )

        save_next_button.setEnabled(False)

        update_save_button_text()
        update_image_labels()
        update_landmark_labels()

        instruction.setText(
            "Choose cusp count, then click Start Digitizing."
        )

    def load_current_image() -> None:
        image = imread(
            str(session.image_path)
        )

        image_layer.data = image

        image_layer.name = (
            f"tooth: {session.image_path.name}"
        )

        try:
            image_layer.reset_contrast_limits()
        except Exception:
            pass

        reset_for_current_image()

        try:
            viewer.reset_view()
        except Exception:
            pass

        print(
            f"Loaded {session.image_progress}: "
            f"{session.image_path}"
        )

    @magicgui(
        call_button="Start Digitizing",
        cusp_count={
            "choices": [
                "1",
                "2",
                "2-pos",
                "3",
                "4",
                "5",
            ],
            "label": "Cusps",
        },
    )
    def start_widget(
        cusp_count: str = "3",
    ):
        try:
            session.landmark_names = (
                load_landmark_names(
                    cusp_count
                )
            )

        except Exception as exc:
            instruction.setText(
                f"Error: {exc}"
            )
            return

        session.clicked_points.clear()
        session.current_index = 0
        session.active = True

        session.out_path = find_output_path(
            session.image_path
        )

        points_layer.data = []
        points_layer.mode = "add"

        save_next_button.setEnabled(False)

        status_label.setText(
            "Status: digitizing"
        )

        update_landmark_labels()

        instruction.setText(
            "Click landmarks in order. "
            "Use Undo if needed. "
            "When complete, click Save & Next Tooth."
        )

    layout.addWidget(
        start_widget.native
    )

    undo_button = QPushButton(
        "Undo Last Point"
    )

    undo_button.setShortcut(
        QKeySequence("Ctrl+Z")
    )

    layout.addWidget(
        undo_button
    )

    layout.addWidget(
        save_next_button
    )

    layout.addWidget(
        skip_button
    )

    def on_points_change(event=None) -> None:
        if (
            not session.active
            or not session.landmark_names
        ):
            return

        coordinates = list(
            points_layer.data
        )

        if len(coordinates) <= len(
            session.clicked_points
        ):
            return

        if len(
            session.clicked_points
        ) >= session.total_landmarks:
            points_layer.data = (
                points_layer.data[
                    : session.total_landmarks
                ]
            )
            return

        # Napari stores 2D points as (y, x).
        new_point: tuple[float, float] = (
            float(coordinates[-1][0]),
            float(coordinates[-1][1]),
        )

        session.clicked_points.append(
            new_point
        )

        session.current_index = len(
            session.clicked_points
        )

        update_landmark_labels()

        if session.is_complete:
            points_layer.mode = "pan_zoom"

            save_next_button.setEnabled(
                True
            )

            status_label.setText(
                "Status: ready to save"
            )

            instruction.setText(
                "All landmarks recorded. "
                "Review the points, use Undo if needed, "
                "then click Save & Next Tooth."
            )

    def undo_last_point() -> None:
        if (
            not session.active
            or not session.clicked_points
        ):
            instruction.setText(
                "Nothing to undo yet."
            )
            return

        session.clicked_points.pop()

        session.current_index = len(
            session.clicked_points
        )

        points_layer.data = (
            points_layer.data[:-1]
        )

        points_layer.mode = "add"

        save_next_button.setEnabled(
            False
        )

        status_label.setText(
            "Status: digitizing"
        )

        update_landmark_labels()

        instruction.setText(
            "Undid last point. "
            "Continue with the current landmark."
        )

    def advance_to_next_image(
        action_message: str,
    ) -> None:
        upcoming = next_pending_index(
            session.image_paths,
            session.image_index + 1,
        )

        if upcoming is None:
            session.active = False
            points_layer.mode = "pan_zoom"

            save_next_button.setEnabled(
                False
            )

            skip_button.setEnabled(
                False
            )

            instruction.setText(
                f"{action_message} "
                "The image queue is complete; "
                "close Napari when ready."
            )

            print(
                "Finished all saved-or-skipped "
                "images in the queue."
            )

            return

        session.image_index = upcoming

        try:
            load_current_image()

        except Exception as exc:
            session.active = False

            save_next_button.setEnabled(
                False
            )

            skip_button.setEnabled(
                False
            )

            instruction.setText(
                f"{action_message}, but could not "
                f"load the next image: {exc}"
            )

            print(
                f"Could not load "
                f"{session.image_path}: {exc}"
            )

    def save_and_advance() -> None:
        if not session.is_complete:
            instruction.setText(
                f"Cannot save yet: "
                f"{len(session.clicked_points)} of "
                f"{session.total_landmarks} "
                "landmarks are recorded."
            )
            return

        try:
            save_session(
                session
            )

        except Exception as exc:
            instruction.setText(
                f"Could not save this tooth: {exc}"
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

    def skip_current_image() -> None:
        if session.clicked_points:
            response = QMessageBox.question(
                root,
                "Skip this tooth?",
                (
                    "This will discard the unsaved landmarks "
                    "for the current image, record the image "
                    "in skipped_files.csv, and move to the "
                    "next tooth."
                ),
                (
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No
                ),
                QMessageBox.StandardButton.No,
            )

            if (
                response
                != QMessageBox.StandardButton.Yes
            ):
                return

        skipped_image = session.image_path

        try:
            record_skipped_file(
                skipped_image
            )

        except Exception as exc:
            instruction.setText(
                f"Could not record skipped image: {exc}"
            )
            return

        session.clear_landmarks()

        points_layer.data = []
        points_layer.mode = "pan_zoom"

        save_next_button.setEnabled(
            False
        )

        status_label.setText(
            f"Status: skipped "
            f"{skipped_image.name}"
        )

        advance_to_next_image(
            "Skipped successfully."
        )

    undo_button.clicked.connect(
        undo_last_point
    )

    save_next_button.clicked.connect(
        save_and_advance
    )

    skip_button.clicked.connect(
        skip_current_image
    )

    points_layer.events.data.connect(
        on_points_change
    )

    update_image_labels()
    update_landmark_labels()
    update_save_button_text()

    return root


def main() -> None:
    try:
        image_paths = load_image_queue()

    except Exception as exc:
        print(
            f"Could not load image queue: {exc}"
        )
        return

    start_index = next_pending_index(
        image_paths,
        0,
    )

    if start_index is None:
        print(
            "Every image in the queue has "
            "already been saved or skipped."
        )
        return

    session = DigitizeSession(
        image_paths=image_paths,
        image_index=start_index,
    )

    try:
        initial_image = imread(
            str(session.image_path)
        )

    except Exception as exc:
        print(
            "Could not open the first unsaved image: "
            f"{session.image_path}"
        )
        print(exc)
        return

    viewer = napari.Viewer()

    image_layer = viewer.add_image(
        initial_image,
        name=f"tooth: {session.image_path.name}",
    )

    points_layer = getattr(
        viewer,
        "add_points",
    )(
        name="landmarks",
        face_color="red",
        size=8,
    )

    widget = build_control_widget(
        session=session,
        viewer=viewer,
        image_layer=image_layer,
        points_layer=points_layer,
    )

    viewer.window.add_dock_widget(
        widget,
        area="right",
    )

    saved_count = sum(
        has_saved_output(path)
        for path in image_paths
    )

    skipped_paths = load_skipped_paths()

    skipped_count = sum(
        path_key(path) in skipped_paths
        for path in image_paths
    )

    print(
        f"Loaded {len(image_paths)} unique image(s) "
        f"from {QUEUE_FILE}"
    )

    print(
        f"Already saved: {saved_count}"
    )

    print(
        f"Already skipped: {skipped_count}"
    )

    print(
        f"Starting at {session.image_progress}: "
        f"{session.image_path}"
    )

    print(
        "Choose cusp count in the right-side panel, "
        "then start digitizing."
    )

    napari.run()

    skipped_paths = load_skipped_paths()

    remaining_count = sum(
        not has_saved_output(path)
        and path_key(path) not in skipped_paths
        for path in image_paths
    )

    print(
        f"Session closed. "
        f"Images still pending: {remaining_count}"
    )


if __name__ == "__main__":
    main()
