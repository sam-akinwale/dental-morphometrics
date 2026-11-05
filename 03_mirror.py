from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent
EXPORTS_DIR = BASE / "exports"


# ============================================================
# ORIENT ONE TOOTH
# ============================================================

def mirror_if_needed(df):
    """
    Standardize anatomical orientation so that:

        anterior = left  = negative x
        posterior = right = positive x

    The rule uses Vant and Vpost:

        Vant.x < Vpost.x
            -> already correct

        Vant.x > Vpost.x
            -> mirror across y-axis

    Mirroring is:

        x_new = -x_old
        y_new = y_old

    Step 2 should already guarantee:
        - BA-BP horizontal
        - cusp A above baseline

    This step does NOT:
        - translate
        - rotate
        - scale
        - change y coordinates
    """

    required = {"landmark", "x", "y"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain these columns: {required}"
        )

    vant_rows = df[
        df["landmark"] == "Vant"
    ]

    vpost_rows = df[
        df["landmark"] == "Vpost"
    ]

    if len(vant_rows) != 1:
        raise ValueError(
            f"Expected exactly one Vant, "
            f"found {len(vant_rows)}"
        )

    if len(vpost_rows) != 1:
        raise ValueError(
            f"Expected exactly one Vpost, "
            f"found {len(vpost_rows)}"
        )

    vant_x = float(
        vant_rows.iloc[0]["x"]
    )

    vpost_x = float(
        vpost_rows.iloc[0]["x"]
    )

    oriented = df.copy()

    mirrored = False

    # --------------------------------------------------------
    # If anterior is currently on the RIGHT,
    # reflect across the y-axis.
    # --------------------------------------------------------

    if vant_x > vpost_x:

        oriented["x"] = (
            -oriented["x"]
        )

        mirrored = True

    return oriented, mirrored


# ============================================================
# QC SVG
# ============================================================

def save_svg(
    df,
    output_path,
    mirrored,
):
    """
    Save a QC SVG after Step 3.

    Expected final orientation:

        LEFT                RIGHT
        anterior            posterior

        Vant ... A ... Vpost

        BA ---- (0,0) ---- BP

    Note:
    BA and BP themselves are NOT being used to determine
    anterior/posterior. Vant and Vpost are.
    """

    x = (
        df["x"]
        .astype(float)
        .to_numpy()
    )

    y = (
        df["y"]
        .astype(float)
        .to_numpy()
    )

    labels = (
        df["landmark"]
        .astype(str)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Flip y FOR DISPLAY ONLY.
    #
    # CSV still uses image-coordinate convention:
    # negative y = visually above baseline
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

    # ========================================================
    # COORDINATE AXES
    # ========================================================

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

    # ========================================================
    # BA-BP BASELINE
    # ========================================================

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

    # ========================================================
    # CROWN OUTLINE
    # ========================================================

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
            f"{sx(px):.2f},"
            f"{sy(py):.2f}"
            for px, py in outline
        )

        elements.append(
            f'<polyline '
            f'points="{polyline}" '
            f'fill="none" '
            f'stroke="#777777" '
            f'stroke-width="2"/>'
        )

    # ========================================================
    # LANDMARKS
    # ========================================================

    for xi, yi, label in zip(
        x,
        display_y,
        labels,
    ):

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

    # ========================================================
    # ORIGIN
    # ========================================================

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

    # ========================================================
    # ANATOMICAL DIRECTION
    # ========================================================

    elements.append(
        '<text '
        'x="20" '
        'y="30" '
        'font-size="16" '
        'font-family="Arial" '
        'font-weight="bold">'
        '← ANTERIOR'
        '</text>'
    )

    elements.append(
        f'<text '
        f'x="{svg_width - 145}" '
        f'y="30" '
        f'font-size="16" '
        f'font-family="Arial" '
        f'font-weight="bold">'
        f'POSTERIOR →'
        f'</text>'
    )

    # ========================================================
    # MIRROR STATUS
    # ========================================================

    if mirrored:

        status = (
            "Mirrored across y-axis"
        )

    else:

        status = (
            "Already anterior-left"
        )

    elements.append(
        f'<text '
        f'x="20" '
        f'y="{svg_height - 20}" '
        f'font-size="12" '
        f'font-family="Arial">'
        f'{status}'
        f'</text>'
    )

    # ========================================================
    # BUILD SVG
    # ========================================================

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

    oriented, mirrored = (
        mirror_if_needed(df)
    )

    specimen_name = (
        csv_path.stem.replace(
            "_2_rotated",
            "",
        )
    )

    output_csv = (
        csv_path.parent
        / f"{specimen_name}_3_mirrored.csv"
    )

    output_svg = (
        csv_path.parent
        / f"{specimen_name}_3_mirrored.svg"
    )

    oriented.to_csv(
        output_csv,
        index=False,
    )

    save_svg(
        oriented,
        output_svg,
        mirrored,
    )

    # ========================================================
    # QC CHECKS
    # ========================================================

    vant = oriented[
        oriented["landmark"] == "Vant"
    ].iloc[0]

    vpost = oriented[
        oriented["landmark"] == "Vpost"
    ].iloc[0]

    a = oriented[
        oriented["landmark"] == "A"
    ].iloc[0]

    ba = oriented[
        oriented["landmark"] == "BA"
    ].iloc[0]

    bp = oriented[
        oriented["landmark"] == "BP"
    ].iloc[0]

    anterior_left = (
        float(vant["x"])
        < float(vpost["x"])
    )

    a_above = (
        float(a["y"]) < 0
    )

    baseline_dy = abs(
        float(
            bp["y"]
            - ba["y"]
        )
    )

    status = (
        "MIRRORED"
        if mirrored
        else
        "unchanged"
    )

    print(
        f"✓ {csv_path.name}"
        f" | {status}"
        f" | anterior-left = "
        f"{anterior_left}"
        f" | A above = "
        f"{a_above}"
        f" | baseline Δy = "
        f"{baseline_dy:.8f}"
    )

    return mirrored


# ============================================================
# MAIN
# ============================================================

def main():

    csv_files = sorted(
        EXPORTS_DIR.rglob(
            "*_2_rotated.csv"
        )
    )

    print()
    print(
        f"Found {len(csv_files)} "
        f"Step-2 teeth."
    )
    print()

    successful = 0
    failed = 0
    mirrored_count = 0

    for csv_path in csv_files:

        try:

            mirrored = (
                process_file(csv_path)
            )

            successful += 1

            if mirrored:

                mirrored_count += 1

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
        "STEP 3 COMPLETE"
    )
    print(
        "-----------------------------"
    )

    print(
        f"Successful:       "
        f"{successful}"
    )

    print(
        f"Mirrored:         "
        f"{mirrored_count}"
    )

    print(
        f"Already correct:  "
        f"{successful - mirrored_count}"
    )

    print(
        f"Failed:           "
        f"{failed}"
    )

    print()


if __name__ == "__main__":
    main()
