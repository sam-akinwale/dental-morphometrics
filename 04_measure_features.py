from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent

EXPORTS_DIR = BASE / "exports"
RESULTS_DIR = BASE / "results"
METADATA_DIR = BASE / "metadata"

RESULTS_DIR.mkdir(exist_ok=True)


OUTPUT_FILE = (
    RESULTS_DIR
    / "tooth_features.csv"
)


FEATURES_METADATA_FILE = (
    RESULTS_DIR
    / "tooth_features_with_metadata.csv"
)


# ============================================================
# FIND MOST RECENT MASTER DATASET
# ============================================================

master_files = list(
    METADATA_DIR.glob(
        "tooth_metadata_*_with_features.csv"
    )
)


def extract_tooth_count(path):
    """
    Extract the tooth count from filenames such as:

        tooth_metadata_123_with_features.csv
    """

    match = re.search(
        r"tooth_metadata_(\d+)_with_features\.csv$",
        path.name,
    )

    return (
        int(match.group(1))
        if match
        else -1
    )


MASTER_FILE = (
    max(
        master_files,
        key=extract_tooth_count,
    )
    if master_files
    else None
)


if MASTER_FILE is not None:

    print(
        f"Using master dataset: "
        f"{MASTER_FILE.name}"
    )


# ============================================================
# BASIC GEOMETRY
# ============================================================

def point(df, name):
    """
    Return x,y coordinates for one landmark.

    Returns None unless exactly one landmark with the
    requested name is present.
    """

    rows = df.loc[
        df["landmark"] == name
    ]

    if len(rows) != 1:
        return None

    row = rows.iloc[0]

    return np.array(
        [
            float(row["x"]),
            float(row["y"]),
        ]
    )


def distance(p1, p2):
    """
    Euclidean distance between two points.
    """

    if (
        p1 is None
        or p2 is None
    ):
        return np.nan

    return float(
        np.linalg.norm(
            p2 - p1
        )
    )


def point_to_line_distance(
    p,
    a,
    b,
):
    """
    Shortest perpendicular distance from point p
    to the infinite line passing through a and b.

    Used for global cusp height relative to the
    BA-BP tooth baseline.
    """

    if (
        p is None
        or a is None
        or b is None
    ):
        return np.nan

    ab = b - a

    denominator = (
        np.linalg.norm(ab)
    )

    if denominator == 0:
        return np.nan

    return float(
        abs(
            np.cross(
                ab,
                p - a,
            )
        )
        / denominator
    )


def angle_three_points(
    p1,
    vertex,
    p2,
):
    """
    Angle p1-vertex-p2 in degrees.

    Used for the B-A-C angle, with A as the vertex.
    """

    if (
        p1 is None
        or vertex is None
        or p2 is None
    ):
        return np.nan

    v1 = p1 - vertex
    v2 = p2 - vertex

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if (
        norm1 == 0
        or norm2 == 0
    ):
        return np.nan

    cos_angle = (
        np.dot(v1, v2)
        / (norm1 * norm2)
    )

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                cos_angle
            )
        )
    )


def safe_ratio(
    numerator,
    denominator,
):
    """
    Divide while safely handling missing or zero values.
    """

    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    return (
        numerator
        / denominator
    )


# ============================================================
# SPECIMEN EXTRACTION
# ============================================================

def extract_specimen(folder_name):
    """
    Extract the specimen identifier from an export-folder name.

    Example:

        arctocephalus_pusillus_bmnh_1925_1_2_68

    becomes:

        bmnh_1925_1_2_68
    """

    folder_name = str(
        folder_name
    ).strip()

    folder_lower = (
        folder_name.lower()
    )


    museum_codes = [
        "bmnh",
        "usnm",
        "amnh",
        "fmnh",
        "rmnh",
        "mnhn",
        "zmb",
        "nhm",
        "nms",
        "nmsz",
        "smf",
        "umz",
        "ummz",
        "yale",
        "mcgill",
    ]


    for museum in museum_codes:

        marker = (
            f"_{museum}_"
        )

        if marker in folder_lower:

            start = (
                folder_lower.index(
                    marker
                )
                + 1
            )

            return (
                folder_name[
                    start:
                ]
                .strip()
                .lower()
            )


        marker_start = (
            f"_{museum}"
        )

        if marker_start in folder_lower:

            start = (
                folder_lower.index(
                    marker_start
                )
                + 1
            )

            return (
                folder_name[
                    start:
                ]
                .strip()
                .lower()
            )


    # --------------------------------------------------------
    # Fallback:
    # assume first two components are genus + species
    # --------------------------------------------------------

    parts = (
        folder_name.split("_")
    )


    if len(parts) > 2:

        return (
            "_".join(
                parts[2:]
            )
            .strip()
            .lower()
        )


    return (
        folder_name
        .strip()
        .lower()
    )


# ============================================================
# ACCESSORY-CUSP RELIEF DEFINITIONS
#
# Relief is the absolute VERTICAL y-distance between each
# accessory cusp tip and its preceding valley.
#
# B -> V(B-A)
# C -> V(A-C)
# D -> V(C-D)
# E -> V(D-E)
#
# A relief is intentionally NOT calculated.
# ============================================================

RELIEF_VALLEYS = {

    "B":
        "V(B-A)",

    "C":
        "V(A-C)",

    "D":
        "V(C-D)",

    "E":
        "V(D-E)",
}


# ============================================================
# MEASURE ONE TOOTH
# ============================================================

def measure_tooth(csv_path):
    """
    Calculate retained morphometric features for one tooth.
    """

    csv_path = Path(
        csv_path
    )


    df = pd.read_csv(
        csv_path
    )


    # --------------------------------------------------------
    # Check required landmark-table columns
    # --------------------------------------------------------

    required_columns = {
        "landmark",
        "x",
        "y",
    }


    missing_columns = (
        required_columns
        - set(df.columns)
    )


    if missing_columns:

        raise ValueError(
            f"{csv_path.name} is missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
        )


    # ========================================================
    # LANDMARKS
    # ========================================================

    BA = point(
        df,
        "BA",
    )


    BP = point(
        df,
        "BP",
    )


    A = point(
        df,
        "A",
    )


    B = point(
        df,
        "B",
    )


    C = point(
        df,
        "C",
    )


    D = point(
        df,
        "D",
    )


    E = point(
        df,
        "E",
    )


    cusp_points = {

        "A": A,

        "B": B,

        "C": C,

        "D": D,

        "E": E,
    }


    # ========================================================
    # METADATA
    # ========================================================

    folder_name = (
        csv_path.parent.name
    )


    specimen = (
        extract_specimen(
            folder_name
        )
    )


    image_name = (
        csv_path.stem
        .removesuffix(
            "_3_mirrored"
        )
    )


    # ========================================================
    # CUSP COUNT
    # ========================================================

    landmarks_present = set(
        df[
            "landmark"
        ]
        .astype(str)
    )


    cusp_count = sum(
        cusp in landmarks_present

        for cusp in [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]
    )


    # ========================================================
    # INITIAL OUTPUT
    # ========================================================

    result = {

        "image_name":
            image_name,

        "specimen":
            specimen,

        "cusp_count":
            cusp_count,
    }


    # ========================================================
    # 1. BASELINE LENGTH
    #
    # Euclidean distance:
    #
    # BA -> BP
    # ========================================================

    baseline_length = distance(
        BA,
        BP,
    )


    result[
        "baseline_length"
    ] = baseline_length


    # ========================================================
    # 2. GLOBAL CUSP HEIGHTS
    #
    # Perpendicular distance from each cusp apex
    # to the BA-BP baseline.
    # ========================================================

    global_heights = {}


    for (
        cusp,
        cusp_point,
    ) in cusp_points.items():

        height = (
            point_to_line_distance(
                cusp_point,
                BA,
                BP,
            )
        )


        global_heights[
            cusp
        ] = height


        result[
            f"global_height_{cusp}"
        ] = height


    # ========================================================
    # 3. ACCESSORY-CUSP VERTICAL RELIEF
    #
    # B:
    #   | y_B - y_V(B-A) |
    #
    # C:
    #   | y_C - y_V(A-C) |
    #
    # D:
    #   | y_D - y_V(C-D) |
    #
    # E:
    #   | y_E - y_V(D-E) |
    #
    # A relief is NOT calculated.
    # ========================================================

    local_reliefs = {}


    for (
        cusp,
        valley_name,
    ) in RELIEF_VALLEYS.items():

        cusp_point = point(
            df,
            cusp,
        )


        valley_point = point(
            df,
            valley_name,
        )


        if (
            cusp_point is None
            or valley_point is None
        ):

            relief = np.nan


        else:

            relief = abs(
                float(
                    cusp_point[1]
                    - valley_point[1]
                )
            )


        local_reliefs[
            cusp
        ] = relief


        result[
            f"local_relief_{cusp}"
        ] = relief


    # ========================================================
    # 4. TOTAL ACCESSORY RELIEF RELATIVE TO A
    #
    # Formula:
    #
    #                   R_B + R_C + R_D + R_E
    # TAR = 1 + -----------------------------------
    #                            H_A
    #
    # where:
    #
    # H_A = global height of A
    #
    # R_B ... R_E = vertical accessory-cusp relief
    #
    # Absent cusps contribute zero.
    #
    # A cusp that is PRESENT but has a missing relief
    # measurement causes the metric to be NaN.
    #
    # A-only tooth = 1.0
    # ========================================================

    accessory_cusps = [
        "B",
        "C",
        "D",
        "E",
    ]


    missing_present_relief = any(
        (
            cusp in landmarks_present
            and pd.isna(
                local_reliefs.get(
                    cusp,
                    np.nan,
                )
            )
        )
        for cusp in accessory_cusps
    )


    if missing_present_relief:

        total_accessory_relief_relative_to_A = (
            np.nan
        )


    else:

        total_accessory_relief = sum(
            (
                local_reliefs.get(
                    cusp,
                    0.0,
                )
                if cusp in landmarks_present
                else 0.0
            )
            for cusp in accessory_cusps
        )


        relief_ratio = safe_ratio(
            total_accessory_relief,
            global_heights.get(
                "A",
                np.nan,
            ),
        )


        if pd.isna(
            relief_ratio
        ):

            total_accessory_relief_relative_to_A = (
                np.nan
            )


        else:

            total_accessory_relief_relative_to_A = (
                1.0
                + relief_ratio
            )


    result[
        "TAR"
    ] = (
        total_accessory_relief_relative_to_A
    )


    # ========================================================
    # 5. TOTAL CUSP HEIGHT AND PROMINENCE (TCHP)
    #
    # Formula:
    #
    #             H_B + H_C + H_D + H_E
    #             + R_B + R_C + R_D + R_E
    # TCHP = 1 + -------------------------------
    #                         H_A
    #
    # where:
    #
    # H_A = global height of A
    #
    # H_B ... H_E = global accessory-cusp heights
    #
    # R_B ... R_E = vertical accessory-cusp relief
    #
    # Absent accessory cusps contribute zero.
    #
    # If a cusp is PRESENT but either its global height
    # or relief is missing, TCHP is NaN.
    #
    # A-only tooth = 1.0
    # ========================================================

    missing_present_tchp_component = any(
        (
            cusp in landmarks_present
            and (
                pd.isna(
                    global_heights.get(
                        cusp,
                        np.nan,
                    )
                )
                or
                pd.isna(
                    local_reliefs.get(
                        cusp,
                        np.nan,
                    )
                )
            )
        )
        for cusp in accessory_cusps
    )


    if missing_present_tchp_component:

        TCHP = np.nan


    else:

        total_accessory_height = sum(
            (
                global_heights.get(
                    cusp,
                    0.0,
                )
                if cusp in landmarks_present
                else 0.0
            )
            for cusp in accessory_cusps
        )


        total_accessory_relief = sum(
            (
                local_reliefs.get(
                    cusp,
                    0.0,
                )
                if cusp in landmarks_present
                else 0.0
            )
            for cusp in accessory_cusps
        )


        tchp_numerator = (
            total_accessory_height
            + total_accessory_relief
        )


        tchp_ratio = safe_ratio(
            tchp_numerator,
            global_heights.get(
                "A",
                np.nan,
            ),
        )


        if pd.isna(
            tchp_ratio
        ):

            TCHP = np.nan


        else:

            TCHP = (
                1.0
                + tchp_ratio
            )


    result[
        "TCHP"
    ] = TCHP


    # ========================================================
    # 6. B-A-C ANGLE
    #
    # A is the vertex.
    # ========================================================

    bac_angle = (
        angle_three_points(
            B,
            A,
            C,
        )
    )


    result[
        "angle_B_A_C_deg"
    ] = bac_angle


    # ========================================================
    # 7. A HEIGHT RELATIVE TO BASELINE WIDTH
    #
    # global_height_A / baseline_length
    # ========================================================

    result[
        "relative_height_A_to_baseline_width"
    ] = safe_ratio(
        global_heights.get(
            "A",
            np.nan,
        ),
        baseline_length,
    )


    # ========================================================
    # 8. ACCESSORY-CUSP HEIGHT RELATIVE TO A
    #
    # B/A
    # C/A
    # D/A
    # E/A
    # ========================================================

    for cusp in [
        "B",
        "C",
        "D",
        "E",
    ]:

        result[
            f"height_{cusp}_relative_to_A"
        ] = safe_ratio(
            global_heights.get(
                cusp,
                np.nan,
            ),
            global_heights.get(
                "A",
                np.nan,
            ),
        )


    return result


# ============================================================
# HELPER:
# UPDATE EXISTING DATAFRAME ROW
# ============================================================

def update_dataframe_row(
    dataframe,
    match,
    measurements,
):
    """
    Replace columns produced by measure_tooth() that already
    exist in an existing dataframe.

    Existing manually curated metadata are preserved.
    """

    updated_columns = []


    for (
        column,
        value,
    ) in measurements.items():

        if column in dataframe.columns:

            dataframe.loc[
                match,
                column,
            ] = value


            updated_columns.append(
                column
            )


    return updated_columns


# ============================================================
# UPDATE ONE CORRECTED TOOTH
# ============================================================

def update_one_tooth(
    csv_path,
):

    csv_path = Path(
        csv_path
    ).resolve()


    if not csv_path.exists():

        raise FileNotFoundError(
            "Landmark CSV not found:\n"
            f"{csv_path}"
        )


    print()
    print("=" * 65)
    print(
        "RECALCULATING ONE CORRECTED TOOTH"
    )
    print("=" * 65)

    print()
    print(
        csv_path
    )
    print()


    new = measure_tooth(
        csv_path
    )


    image_id = new[
        "image_name"
    ]


    print(
        f"Image ID:  {image_id}"
    )


    print(
        f"Specimen:  "
        f"{new['specimen']}"
    )


    print()
    print(
        "KEY RECALCULATED VALUES"
    )
    print("-" * 45)


    print(
        f"Cusp count: "
        f"{new['cusp_count']}"
    )


    print(
        f"A global height: "
        f"{new['global_height_A']:.6f}"
    )


    print(
        "A height / baseline width: "
        f"{new['relative_height_A_to_baseline_width']:.6f}"
    )


    print(
        "Total accessory relief / A: "
        f"{new["TAR"]:.6f}"
    )


    print(
        f"TCHP: "
        f"{new['TCHP']:.6f}"
    )


    for cusp in [
        "B",
        "C",
        "D",
        "E",
    ]:

        relief = new.get(
            f"local_relief_{cusp}",
            np.nan,
        )


        print(
            f"{cusp} vertical relief: "
            f"{relief:.6f}"
        )


    # ========================================================
    # UPDATE tooth_features.csv
    # ========================================================

    print()
    print(
        "Updating tooth_features.csv..."
    )


    if OUTPUT_FILE.exists():

        features = pd.read_csv(
            OUTPUT_FILE
        )


        if (
            "image_name"
            not in features.columns
        ):

            raise ValueError(
                "tooth_features.csv has "
                "no image_name column."
            )


        match = (
            features[
                "image_name"
            ]
            .astype(str)
            .eq(
                image_id
            )
        )


        number_matches = int(
            match.sum()
        )


        if number_matches == 1:

            updated = (
                update_dataframe_row(
                    features,
                    match,
                    new,
                )
            )


            features.to_csv(
                OUTPUT_FILE,
                index=False,
            )


            print(
                "✓ tooth_features.csv updated"
            )


            print(
                f"  {len(updated)} "
                "columns replaced"
            )


        elif number_matches == 0:

            print(
                "WARNING: "
                f"{image_id} was not found."
            )


        else:

            raise ValueError(
                f"Found {number_matches} "
                f"rows for {image_id} "
                "in tooth_features.csv."
            )


    else:

        print(
            "WARNING: "
            "tooth_features.csv does not exist."
        )


    # ========================================================
    # UPDATE tooth_features_with_metadata.csv
    # ========================================================

    print()
    print(
        "Updating "
        "tooth_features_with_metadata.csv..."
    )


    if FEATURES_METADATA_FILE.exists():

        features_meta = (
            pd.read_csv(
                FEATURES_METADATA_FILE
            )
        )


        if (
            "image_name"
            not in features_meta.columns
        ):

            raise ValueError(
                "tooth_features_with_metadata.csv "
                "has no image_name column."
            )


        match = (
            features_meta[
                "image_name"
            ]
            .astype(str)
            .eq(
                image_id
            )
        )


        number_matches = int(
            match.sum()
        )


        if number_matches == 1:

            updated = (
                update_dataframe_row(
                    features_meta,
                    match,
                    new,
                )
            )


            features_meta.to_csv(
                FEATURES_METADATA_FILE,
                index=False,
            )


            print(
                "✓ tooth_features_with_metadata.csv "
                "updated"
            )


            print(
                f"  {len(updated)} "
                "columns replaced"
            )


        elif number_matches == 0:

            print(
                "WARNING: "
                f"{image_id} was not found."
            )


        else:

            raise ValueError(
                f"Found {number_matches} "
                f"rows for {image_id} in "
                "tooth_features_with_metadata.csv."
            )


    else:

        print(
            "NOTE: "
            "tooth_features_with_metadata.csv "
            "does not exist."
        )


    # ========================================================
    # UPDATE MASTER ANALYTICAL DATASET
    # ========================================================

    print()
    print(
        "Updating master analytical dataset..."
    )


    if (
        MASTER_FILE is not None
        and MASTER_FILE.exists()
    ):

        master = pd.read_csv(
            MASTER_FILE
        )


        if (
            "image_id"
            not in master.columns
        ):

            raise ValueError(
                "Master dataset has "
                "no image_id column."
            )


        match = (
            master[
                "image_id"
            ]
            .astype(str)
            .eq(
                image_id
            )
        )


        number_matches = int(
            match.sum()
        )


        if number_matches == 1:

            master_measurements = {

                key: value

                for key, value
                in new.items()

                if key not in {
                    "image_name",
                    "specimen",
                }
            }


            updated = (
                update_dataframe_row(
                    master,
                    match,
                    master_measurements,
                )
            )


            master.to_csv(
                MASTER_FILE,
                index=False,
            )


            print(
                f"✓ {MASTER_FILE.name} updated"
            )


            print(
                f"  {len(updated)} "
                "feature columns replaced"
            )


        elif number_matches == 0:

            print(
                f"NOTE: {image_id} is not "
                "part of the analytical dataset."
            )


            print(
                "Master dataset unchanged."
            )


        else:

            raise ValueError(
                f"Found {number_matches} "
                f"rows for {image_id} "
                "in master dataset."
            )


    else:

        print(
            "NOTE: master analytical "
            "dataset does not exist."
        )


    print()
    print("=" * 65)
    print(
        "SINGLE-TOOTH UPDATE COMPLETE"
    )
    print("=" * 65)

    print()
    print(
        f"Image:     {image_id}"
    )

    print(
        f"Specimen:  "
        f"{new['specimen']}"
    )

    print()


    return new


# ============================================================
# MEASURE ALL TEETH
# ============================================================

def measure_all():

    files = sorted(
        path

        for path
        in EXPORTS_DIR.rglob(
            "*_3_mirrored.csv"
        )

        if (
            "duplicates"
            not in path.relative_to(
                EXPORTS_DIR
            ).parts
        )
    )


    print()
    print(
        f"Found {len(files)} mirrored teeth."
    )
    print()


    results = []
    failed = []


    for csv_path in files:

        try:

            measurements = (
                measure_tooth(
                    csv_path
                )
            )


            results.append(
                measurements
            )


            print(
                f"✓ "
                f"{measurements['image_name']}  "
                f"{measurements['specimen']}"
            )


        except Exception as exc:

            failed.append(
                (
                    csv_path,
                    str(exc),
                )
            )


            print(
                f"✗ {csv_path.name}"
            )


            print(
                f"  {exc}"
            )


    # --------------------------------------------------------
    # Build new feature table from scratch
    # --------------------------------------------------------

    output = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Preferred column order
    # --------------------------------------------------------

    preferred_columns = [

        "image_name",
        "specimen",
        "cusp_count",
        
        "TAR",

        "TCHP",

        "baseline_length",

        "global_height_A",
        "global_height_B",
        "global_height_C",
        "global_height_D",
        "global_height_E",

        "local_relief_B",
        "local_relief_C",
        "local_relief_D",
        "local_relief_E",

        "angle_B_A_C_deg",

        "relative_height_A_to_baseline_width",

        "height_B_relative_to_A",
        "height_C_relative_to_A",
        "height_D_relative_to_A",
        "height_E_relative_to_A",
    ]


    preferred_columns = [
        column

        for column
        in preferred_columns

        if column in output.columns
    ]


    output = output[
        preferred_columns
    ]


    # --------------------------------------------------------
    # QC before saving
    # --------------------------------------------------------

    if output[
        "image_name"
    ].duplicated().any():

        duplicates = output.loc[
            output[
                "image_name"
            ].duplicated(
                keep=False
            )
        ]


        print()
        print(
            "DUPLICATE IMAGE IDs:"
        )


        print(
            duplicates[
                [
                    "image_name",
                    "specimen",
                ]
            ]
        )


        raise ValueError(
            "Duplicate image IDs detected. "
            "tooth_features.csv was not saved."
        )


    missing_specimen = (
        output[
            "specimen"
        ]
        .isna()
        |
        output[
            "specimen"
        ]
        .astype(str)
        .str.strip()
        .eq("")
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 65)
    print(
        "FEATURE EXTRACTION COMPLETE"
    )
    print("=" * 65)


    print(
        f"Measured:              "
        f"{len(results)}"
    )


    print(
        f"Failed:                "
        f"{len(failed)}"
    )


    print(
        f"Missing specimen IDs:  "
        f"{missing_specimen.sum()}"
    )


    print(
        f"Output columns:        "
        f"{len(output.columns)}"
    )


    print()
    print(
        "Saved:"
    )


    print(
        OUTPUT_FILE
    )


    # --------------------------------------------------------
    # Show specimen + composite examples
    # --------------------------------------------------------

    print()
    print(
        "OUTPUT CHECK"
    )
    print("-" * 65)


    print(
        output[
            [
                "image_name",
                "specimen",
                "cusp_count",
                "TAR",
                "TCHP",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


    if failed:

        print()
        print(
            "FAILED FILES:"
        )


        for (
            path,
            reason,
        ) in failed:

            print(
                path
            )

            print(
                f"    {reason}"
            )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate streamlined tooth features "
            "from mirrored landmark CSV files."
        )
    )


    parser.add_argument(
        "--update",
        type=Path,
        default=None,
        help=(
            "Recalculate one corrected "
            "*_3_mirrored.csv and update "
            "that tooth in existing datasets."
        ),
    )


    args = (
        parser.parse_args()
    )


    if (
        args.update
        is not None
    ):

        update_one_tooth(
            args.update
        )


    else:

        measure_all()


if __name__ == "__main__":

    main()