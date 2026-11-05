from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent
EXPORTS_DIR = BASE / "exports"


# ============================================================
# ROTATE ONE TOOTH
# ============================================================

def rotate_to_horizontal_baseline(df):
    """
    Rotate a translated tooth so that:

    1. BA-BP is horizontal.
    2. Cusp A is above the baseline.

    Assumes Step 1 has already placed the BA-BP midpoint
    at (0,0).

    IMPORTANT:
    Coordinates retain Napari/image convention:
        +y = downward

    Therefore, after correct orientation:
        cusp A should have y < 0.

    This changes ORIENTATION ONLY.

    It does NOT:
    - scale
    - mirror
    - change distances
    - change angles
    """

    required = {"landmark", "x", "y"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain these columns: {required}"
        )

    ba_rows = df[df["landmark"] == "BA"]
    bp_rows = df[df["landmark"] == "BP"]
    a_rows = df[df["landmark"] == "A"]

    if len(ba_rows) != 1:
        raise ValueError(
            f"Expected exactly one BA, found {len(ba_rows)}"
        )

    if len(bp_rows) != 1:
        raise ValueError(
            f"Expected exactly one BP, found {len(bp_rows)}"
        )

    if len(a_rows) != 1:
        raise ValueError(
            f"Expected exactly one A, found {len(a_rows)}"
        )

    ba = ba_rows.iloc[0]
    bp = bp_rows.iloc[0]

    # --------------------------------------------------------
    # STEP 2A:
    # Calculate angle of BA -> BP
    # --------------------------------------------------------

    dx = float(bp["x"] - ba["x"])
    dy = float(bp["y"] - ba["y"])

    baseline_length = math.hypot(dx, dy)

    if baseline_length == 0:
        raise ValueError(
            "BA and BP have identical coordinates."
        )

    baseline_angle = math.atan2(dy, dx)

    # Rotate by the negative of that angle
    theta = -baseline_angle

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    rotation_matrix = np.array(
        [
            [cos_t, -sin_t],
            [sin_t,  cos_t],
        ]
    )

    coords = df[["x", "y"]].to_numpy(
        dtype=float
    )

    rotated_coords = (
        coords @ rotation_matrix.T
    )

    rotated = df.copy()

    rotated["x"] = rotated_coords[:, 0]
    rotated["y"] = rotated_coords[:, 1]

    # --------------------------------------------------------
    # STEP 2B:
    # Make sure cusp A is ABOVE the baseline.
    #
    # Because these are image coordinates:
    #
    #       y < 0 = visually above baseline
    #       y > 0 = visually below baseline
    #
    # If A is below the baseline, rotate the ENTIRE tooth
    # another 180 degrees.
    # --------------------------------------------------------

    a_after = rotated[
        rotated["landmark"] == "A"
    ].iloc[0]

    rotated_180 = False

    if float(a_after["y"]) > 0:

        rotated["x"] = -rotated["x"]
        rotated["y"] = -rotated["y"]

        rotated_180 = True

    return (
        rotated,
        math.degrees(baseline_angle),
        rotated_180,
    )


# ============================================================
# QC SVG
# ============================================================

def save_svg(
    df,
    output_path,
    original_angle,
    rotated_180,
):
    """
    Save an XY QC plot of the rotated landmarks.

    All completed teeth should have:

        - horizontal BA-BP baseline
        - BA-BP midpoint at (0,0)
        - cusp A above the baseline

    y is flipped FOR DISPLAY ONLY.
    """

    x = df["x"].astype(float).to_numpy()
    y = df["y"].astype(float).to_numpy()

    labels = (
        df["landmark"]
        .astype(str)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Flip y FOR DISPLAY ONLY
    # --------------------------------------------------------

    display_y = -y

    svg_width = 700
    svg_height = 500

    width = max(
        float(x.max() - x.min()),
        1.0,
    )

    height = max(
        float(
            display_y.max()
            - display_y.min()
        ),
        1.0,
    )

    pad_x = width * 0.20
    pad_y = height * 0.20

    min_x = min(
        float(x.min()) - pad_x,
        -pad_x,
    )

    max_x = max(
        float(x.max()) + pad_x,
        pad_x,
    )

    min_y = min(
        float(display_y.min()) - pad_y,
        -pad_y,
    )

    max_y = max(
        float(display_y.max()) + pad_y,
        pad_y,
    )

    def sx(value):
        return (
            (value - min_x)
            / (max_x - min_x)
            * svg_width
        )

    def sy(value):
        return (
            svg_height
            - (value - min_y)
            / (max_y - min_y)
            * svg_height
        )

    elements = []

    origin_x = sx(0)
    origin_y = sy(0)

    # --------------------------------------------------------
    # Coordinate axes
    # --------------------------------------------------------

    elements.append(
        f'<line '
        f'x1="0" '
        f'y1="{origin_y:.2f}" '
        f'x2="{svg_width}" '
        f'y2="{origin_y:.2f}" '
        f'stroke="#cccccc" '
        f'stroke-width="1"/>'
    )

    elements.append(
        f'<line '
        f'x1="{origin_x:.2f}" '
        f'y1="0" '
        f'x2="{origin_x:.2f}" '
        f'y2="{svg_height}" '
        f'stroke="#cccccc" '
        f'stroke-width="1"/>'
    )

    # --------------------------------------------------------
    # BA-BP baseline
    # --------------------------------------------------------

    ba = df[
        df["landmark"] == "BA"
    ].iloc[0]

    bp = df[
        df["landmark"] == "BP"
    ].iloc[0]

    elements.append(
        f'<line '
        f'x1="{sx(float(ba["x"])):.2f}" '
        f'y1="{sy(-float(ba["y"])):.2f}" '
        f'x2="{sx(float(bp["x"])):.2f}" '
        f'y2="{sy(-float(bp["y"])):.2f}" '
        f'stroke="black" '
        f'stroke-width="3"/>'
    )

    # --------------------------------------------------------
    # Crown outline
    # --------------------------------------------------------

    outline_order = [
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

    point_lookup = {
        row["landmark"]: (
            float(row["x"]),
            -float(row["y"]),
        )
        for _, row in df.iterrows()
    }

    outline = [
        point_lookup[name]
        for name in outline_order
        if name in point_lookup
    ]

    if len(outline) >= 2:

        polyline = " ".join(
            f"{sx(px):.2f},{sy(py):.2f}"
            for px, py in outline
        )

        elements.append(
            f'<polyline '
            f'points="{polyline}" '
            f'fill="none" '
            f'stroke="#777777" '
            f'stroke-width="2"/>'
        )

    # --------------------------------------------------------
    # Landmarks
    # --------------------------------------------------------

    for xi, yi, label in zip(
        x,
        display_y,
        labels,
    ):

        px = sx(float(xi))
        py = sy(float(yi))

        if label in {"BA", "BP"}:
            fill = "#222222"

        elif label.startswith("V"):
            fill = "#377eb8"

        else:
            fill = "#e41a1c"

        elements.append(
            f'<circle '
            f'cx="{px:.2f}" '
            f'cy="{py:.2f}" '
            f'r="5" '
            f'fill="{fill}"/>'
        )

        elements.append(
            f'<text '
            f'x="{px + 7:.2f}" '
            f'y="{py - 7:.2f}" '
            f'font-size="12" '
            f'font-family="Arial">'
            f'{label}'
            f'</text>'
        )

    # --------------------------------------------------------
    # Origin
    # --------------------------------------------------------

    elements.append(
        f'<circle '
        f'cx="{origin_x:.2f}" '
        f'cy="{origin_y:.2f}" '
        f'r="6" '
        f'fill="none" '
        f'stroke="#555555" '
        f'stroke-width="2"/>'
    )

    elements.append(
        f'<text '
        f'x="{origin_x + 8:.2f}" '
        f'y="{origin_y + 18:.2f}" '
        f'font-size="12" '
        f'font-family="Arial">'
        f'(0,0)'
        f'</text>'
    )

    # --------------------------------------------------------
    # QC information
    # --------------------------------------------------------

    correction_text = (
        "Additional 180-degree rotation applied"
        if rotated_180
        else
        "No 180-degree correction required"
    )

    elements.append(
        f'<text '
        f'x="20" '
        f'y="25" '
        f'font-size="12" '
        f'font-family="Arial">'
        f'Original baseline angle: '
        f'{original_angle:.2f} degrees'
        f'</text>'
    )

    elements.append(
        f'<text '
        f'x="20" '
        f'y="45" '
        f'font-size="12" '
        f'font-family="Arial">'
        f'{correction_text}'
        f'</text>'
    )

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>

<svg
    xmlns="http://www.w3.org/2000/svg"
    width="{svg_width}"
    height="{svg_height}"
    viewBox="0 0 {svg_width} {svg_height}"
>

    <rect
        width="100%"
        height="100%"
        fill="white"
    />

    {''.join(elements)}

</svg>
"""

    output_path.write_text(svg)


# ============================================================
# PROCESS ONE TOOTH
# ============================================================

def process_file(csv_path):

    df = pd.read_csv(csv_path)

    (
        rotated,
        original_angle,
        rotated_180,
    ) = rotate_to_horizontal_baseline(df)

    specimen_name = (
        csv_path.stem.replace(
            "_1_translated",
            "",
        )
    )

    output_csv = (
        csv_path.parent
        / f"{specimen_name}_2_rotated.csv"
    )

    output_svg = (
        csv_path.parent
        / f"{specimen_name}_2_rotated.svg"
    )

    # --------------------------------------------------------
    # OVERWRITE existing Step-2 output.
    #
    # This is intentional because Step 2 has now been corrected.
    # Original landmark files and Step-1 files are untouched.
    # --------------------------------------------------------

    rotated.to_csv(
        output_csv,
        index=False,
    )

    save_svg(
        rotated,
        output_svg,
        original_angle,
        rotated_180,
    )

    # --------------------------------------------------------
    # QC CHECKS
    # --------------------------------------------------------

    ba = rotated[
        rotated["landmark"] == "BA"
    ].iloc[0]

    bp = rotated[
        rotated["landmark"] == "BP"
    ].iloc[0]

    a = rotated[
        rotated["landmark"] == "A"
    ].iloc[0]

    baseline_y_difference = abs(
        float(
            bp["y"]
            - ba["y"]
        )
    )

    a_above_baseline = (
        float(a["y"]) < 0
    )

    correction = (
        " + 180°"
        if rotated_180
        else ""
    )

    print(
        f"✓ {csv_path.name}"
        f" | baseline rotation "
        f"{original_angle:.2f}°"
        f"{correction}"
        f" | baseline Δy = "
        f"{baseline_y_difference:.8f}"
        f" | A above = "
        f"{a_above_baseline}"
    )

    return rotated_180


# ============================================================
# MAIN
# ============================================================

def main():

    csv_files = sorted(
        EXPORTS_DIR.rglob(
            "*_1_translated.csv"
        )
    )

    print()
    print(
        f"Found {len(csv_files)} "
        f"translated teeth."
    )
    print()

    successful = 0
    failed = 0
    corrected_180 = 0

    for csv_path in csv_files:

        try:

            rotated_180 = (
                process_file(csv_path)
            )

            successful += 1

            if rotated_180:
                corrected_180 += 1

        except Exception as exc:

            failed += 1

            print(
                f"✗ FAILED: {csv_path}"
            )

            print(
                f"  Reason: {exc}"
            )

    print()
    print(
        "-----------------------------"
    )
    print(
        "STEP 2 COMPLETE"
    )
    print(
        "-----------------------------"
    )

    print(
        f"Successful:             "
        f"{successful}"
    )

    print(
        f"180° corrections:       "
        f"{corrected_180}"
    )

    print(
        f"No 180° correction:      "
        f"{successful - corrected_180}"
    )

    print(
        f"Failed:                 "
        f"{failed}"
    )

    print()


if __name__ == "__main__":
    main()
