from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent
EXPORTS_DIR = BASE / "exports"


# ============================================================
# TRANSLATE ONE TOOTH
# ============================================================

def translate_to_baseline_midpoint(df):
    """
    Translate a tooth so that the midpoint between BA and BP
    is located at (0, 0).

    This changes POSITION ONLY.

    It does NOT:
    - rotate
    - scale
    - mirror
    - alter distances between landmarks
    """

    required = {"landmark", "x", "y"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain these columns: {required}"
        )

    # Find baseline landmarks
    ba_rows = df[df["landmark"] == "BA"]
    bp_rows = df[df["landmark"] == "BP"]

    if len(ba_rows) != 1:
        raise ValueError(
            f"Expected exactly one BA, found {len(ba_rows)}"
        )

    if len(bp_rows) != 1:
        raise ValueError(
            f"Expected exactly one BP, found {len(bp_rows)}"
        )

    ba = ba_rows.iloc[0]
    bp = bp_rows.iloc[0]

    # --------------------------------------------------------
    # Calculate midpoint of BA–BP
    # --------------------------------------------------------

    midpoint_x = (ba["x"] + bp["x"]) / 2
    midpoint_y = (ba["y"] + bp["y"]) / 2

    # --------------------------------------------------------
    # Translate every landmark
    # --------------------------------------------------------

    translated = df.copy()

    translated["x"] = translated["x"] - midpoint_x
    translated["y"] = translated["y"] - midpoint_y

    return translated


# ============================================================
# CREATE SIMPLE QC SVG
# ============================================================

def save_svg(df, output_path):
    """
    Save a simple XY plot of the translated landmarks.

    The origin (0,0) is shown by crosshairs.

    IMPORTANT:
    The baseline is NOT rotated yet.
    """

    x = df["x"].astype(float).to_numpy()
    y = df["y"].astype(float).to_numpy()

    labels = df["landmark"].astype(str).to_numpy()

    # --------------------------------------------------------
    # For display only:
    #
    # Image coordinates have +y downward.
    # Flip y so the SVG looks like a normal tooth.
    #
    # This does NOT change the CSV.
    # --------------------------------------------------------

    display_y = -y

    # Plot dimensions
    svg_width = 700
    svg_height = 500

    # Add padding
    width = max(x.max() - x.min(), 1)
    height = max(display_y.max() - display_y.min(), 1)

    pad_x = width * 0.20
    pad_y = height * 0.20

    min_x = min(x.min() - pad_x, -pad_x)
    max_x = max(x.max() + pad_x, pad_x)

    min_y = min(display_y.min() - pad_y, -pad_y)
    max_y = max(display_y.max() + pad_y, pad_y)

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

    # --------------------------------------------------------
    # X and Y axes through (0,0)
    # --------------------------------------------------------

    origin_x = sx(0)
    origin_y = sy(0)

    elements.append(
        f'<line x1="0" y1="{origin_y:.2f}" '
        f'x2="{svg_width}" y2="{origin_y:.2f}" '
        f'stroke="#cccccc" stroke-width="1"/>'
    )

    elements.append(
        f'<line x1="{origin_x:.2f}" y1="0" '
        f'x2="{origin_x:.2f}" y2="{svg_height}" '
        f'stroke="#cccccc" stroke-width="1"/>'
    )

    # --------------------------------------------------------
    # BA–BP baseline
    # --------------------------------------------------------

    ba = df[df["landmark"] == "BA"].iloc[0]
    bp = df[df["landmark"] == "BP"].iloc[0]

    elements.append(
        f'<line '
        f'x1="{sx(float(ba["x"])):.2f}" '
        f'y1="{sy(-float(ba["y"])):.2f}" '
        f'x2="{sx(float(bp["x"])):.2f}" '
        f'y2="{sy(-float(bp["y"])):.2f}" '
        f'stroke="black" stroke-width="3"/>'
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
            -float(row["y"])
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
            f'<polyline points="{polyline}" '
            f'fill="none" '
            f'stroke="#777777" '
            f'stroke-width="2"/>'
        )

    # --------------------------------------------------------
    # Draw landmarks
    # --------------------------------------------------------

    for xi, yi, label in zip(x, display_y, labels):

        px = sx(float(xi))
        py = sy(float(yi))

        # Baseline endpoints
        if label in {"BA", "BP"}:
            fill = "#222222"

        # Valleys
        elif label.startswith("V"):
            fill = "#377eb8"

        # Cusp apices
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
    # Mark origin
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

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>

<svg xmlns="http://www.w3.org/2000/svg"
     width="{svg_width}"
     height="{svg_height}"
     viewBox="0 0 {svg_width} {svg_height}">

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
# PROCESS ONE CSV
# ============================================================

def process_file(csv_path):

    df = pd.read_csv(csv_path)

    translated = translate_to_baseline_midpoint(df)

    # Remove "_landmarks" from original filename
    #
    # Example:
    #
    # M148369_landmarks.csv
    #
    # becomes
    #
    # M148369_1_translated.csv

    specimen_name = csv_path.stem.replace(
        "_landmarks",
        ""
    )

    output_csv = (
        csv_path.parent
        / f"{specimen_name}_1_translated.csv"
    )

    output_svg = (
        csv_path.parent
        / f"{specimen_name}_1_translated.svg"
    )

    translated.to_csv(
        output_csv,
        index=False
    )

    save_svg(
        translated,
        output_svg
    )

    print(
        f"✓ {csv_path.name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # IMPORTANT:
    # Only grab the ORIGINAL landmark files.
    #
    # This prevents us from accidentally translating
    # translated files later.

    csv_files = sorted(
        EXPORTS_DIR.rglob("*_landmarks.csv")
    )

    print()
    print(
        f"Found {len(csv_files)} teeth."
    )
    print()

    successful = 0
    failed = 0

    for csv_path in csv_files:

        try:

            process_file(csv_path)

            successful += 1

        except Exception as exc:

            failed += 1

            print(
                f"✗ FAILED: {csv_path}"
            )

            print(
                f"  Reason: {exc}"
            )

    print()
    print("-----------------------------")
    print("STEP 1 COMPLETE")
    print("-----------------------------")
    print(
        f"Successful: {successful}"
    )
    print(
        f"Failed:     {failed}"
    )
    print()


if __name__ == "__main__":
    main()
