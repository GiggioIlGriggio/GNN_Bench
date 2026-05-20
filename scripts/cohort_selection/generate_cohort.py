#!/usr/bin/env python3
"""Generate subject-list files for PNC cohort partitioning.

Produces two non-overlapping text files (one subject ID per line)
that can be passed to ``dataset.subject_list_file`` in the Hydra config.

Partitioning strategies
-----------------------
median
    Sort all subjects by age, take the N youngest as partition A and the
    remaining subjects as partition B.

interleaved
    Within each discrete age value, alternate subjects between partitions
    A and B so that both partitions cover (roughly) the same age range.

Arguments
---------
--metadata
    Path to the metadata CSV file (e.g. PNC_ALL_SCORES.csv).
    Must contain subject ID and age columns. Required.

--sc-dir
    Path to the Structural_maps directory containing SC .mat files
    (e.g. sub-XXXXXXXXXX_*_connectome.mat). Required if --modality is
    'sc' or 'both'. Optional otherwise.

--fc-dir
    Path to the Functional_Mats directory containing FC .csv files
    (e.g. sub-XXXXXXXXXX_connectivity_matrix.csv). Required if --modality
    is 'fc' or 'both'. Optional otherwise.

--modality
    Which connectivity modality to consider when filtering subjects.
    - sc: Include only subjects with SC data.
    - fc: Include only subjects with FC data.
    - both: Include only subjects that have both SC and FC data (intersection).
    Default: sc

--age-col
    Name of the column in the metadata CSV that contains participant age.
    Default: age_at_cnb

--id-col
    Name of the column in the metadata CSV that contains subject IDs.
    Default: SUBJID

--pad-width
    Zero-padding width for converting numeric SUBJID to filename format
    sub-XXXXXXXXXX. For PNC, the SUBJID is 12 digits (6000...) and
    filenames use the last 10 digits. Default: 10

--n-young
    Number of subjects to place in partition A (younger/sampled group).
    Default: 130

--strategy
    Partitioning algorithm to use.
    - median: Sort by age, take N youngest → A, rest → B.
    - interleaved: Within each age bin, alternate subjects for matched
      age distribution. Default: median. Required.

--seed
    Random seed for reproducibility (used for tie-breaking in median
    strategy and shuffling within age bins in interleaved strategy).
    Default: 42

--output-dir
    Directory where output files will be written.
    Output filenames: {strategy}_s{seed}_partA_{count}.txt and
    {strategy}_s{seed}_partB_{count}.txt
    Default: scripts/cohort_selection/outputs

Usage
-----
::

    python scripts/cohort_selection/generate_cohort.py \\
        --metadata /media/compa/DATA1/Compa/DATA_DERIVATIVES/PNC/T0/Tabular_data/PNC_ALL_SCORES.csv \\
        --sc-dir   /media/compa/DATA1/Compa/DATA_DERIVATIVES/PNC/T0/Structural_maps \\
        --fc-dir   /media/compa/DATA1/Compa/DATA_DERIVATIVES/PNC/T0/Functional_Mats \\
        --modality both \\
        --age-col  age_at_cnb \\
        --id-col   SUBJID \\
        --pad-width 10 \\
        --n-young  130 \\
        --strategy median \\
        --seed 42 \\
        --output-dir scripts/cohort_selection/outputs

Output files are named ``{strategy}_partA_{n}.txt`` and
``{strategy}_partB_{n}.txt``.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _discover_sc_subjects(sc_dir: Path) -> set[str]:
    """Return the set of ``sub-XXXXXXXXXX`` IDs that have SC .mat files."""
    ids: set[str] = set()
    for mat_path in sc_dir.glob("sub-*_*connectome.mat"):
        m = re.match(r"^(sub-\d+)", mat_path.name)
        if m:
            ids.add(m.group(1))
    return ids


def _discover_fc_subjects(fc_dir: Path) -> set[str]:
    """Return the set of ``sub-XXXXXXXXXX`` IDs that have FC .csv files."""
    ids: set[str] = set()
    for csv_path in fc_dir.glob("sub-*_connectivity_matrix.csv"):
        m = re.match(r"^(sub-\d+)", csv_path.name)
        if m:
            ids.add(m.group(1))
    return ids


def _build_eligible_df(
    metadata: pd.DataFrame,
    id_col: str,
    age_col: str,
    pad_width: int,
    sc_sub_ids: set[str],
    fc_sub_ids: set[str],
    modality: str,
) -> pd.DataFrame:
    """Return a DataFrame of subjects that satisfy the modality requirement.

    Parameters
    ----------
    modality : str
        ``"sc"`` – subjects with SC data only.
        ``"fc"`` – subjects with FC data only.
        ``"both"`` – subjects that have SC **and** FC data.

    Columns: ``sub_id`` (str), ``age`` (float), ``has_sc`` (bool), ``has_fc`` (bool).
    """
    if modality == "sc":
        required = sc_sub_ids
    elif modality == "fc":
        required = fc_sub_ids
    elif modality == "both":
        required = sc_sub_ids & fc_sub_ids
    else:
        raise ValueError(f"Unknown modality '{modality}'. Choose sc | fc | both.")

    rows = []
    for _, row in metadata.iterrows():
        raw_id = str(int(row[id_col]))
        file_id = raw_id[-pad_width:].zfill(pad_width)
        sub_id = f"sub-{file_id}"
        age_val = row.get(age_col)
        if sub_id in required and pd.notna(age_val):
            rows.append({
                "sub_id": sub_id,
                "age": float(age_val),
                "has_sc": sub_id in sc_sub_ids,
                "has_fc": sub_id in fc_sub_ids,
            })
    df = pd.DataFrame(rows).drop_duplicates(subset="sub_id")
    return df.sort_values("age").reset_index(drop=True)


# ------------------------------------------------------------------
# Partitioning strategies
# ------------------------------------------------------------------

def partition_median(
    df: pd.DataFrame,
    n_young: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split by age rank: N youngest → A, rest → B.

    Within ties at the boundary, subjects are randomly assigned using
    ``seed`` for reproducibility.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_rand"] = rng.random(len(df))
    df = df.sort_values(["age", "_rand"]).reset_index(drop=True)

    if n_young > len(df):
        raise ValueError(
            f"Requested n_young={n_young} but only {len(df)} eligible subjects."
        )

    part_a = df.iloc[:n_young]["sub_id"].tolist()
    part_b = df.iloc[n_young:]["sub_id"].tolist()
    return part_a, part_b


def partition_interleaved(
    df: pd.DataFrame,
    n_a: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Bin-balanced split: within each age value, alternate subjects.

    Subjects within each age bin are shuffled (using ``seed``), then
    alternately assigned to partition A or B. Partition A is capped at
    ``n_a`` subjects; overflow goes to B.
    """
    rng = np.random.default_rng(seed)

    if n_a > len(df):
        raise ValueError(
            f"Requested n_a={n_a} but only {len(df)} eligible subjects."
        )

    part_a: list[str] = []
    part_b: list[str] = []

    # Compute how many subjects per age bin go to A (proportional)
    age_bins = df.groupby("age")["sub_id"].apply(list).to_dict()
    total = len(df)

    # Target fraction for partition A
    frac_a = n_a / total

    # For each age bin, assign floor(frac_a * bin_size) to A
    # Collect remainders to distribute extras
    allocation: dict[float, int] = {}
    remainders: list[tuple[float, float]] = []
    allocated_total = 0

    for age_val in sorted(age_bins.keys()):
        bin_size = len(age_bins[age_val])
        exact = frac_a * bin_size
        floor_val = int(np.floor(exact))
        allocation[age_val] = floor_val
        allocated_total += floor_val
        remainders.append((exact - floor_val, age_val))

    # Distribute remaining slots by largest remainder
    extras_needed = n_a - allocated_total
    remainders.sort(key=lambda x: -x[0])
    for _, age_val in remainders[:extras_needed]:
        allocation[age_val] += 1

    # Now assign subjects
    for age_val in sorted(age_bins.keys()):
        subjects = list(age_bins[age_val])
        rng.shuffle(subjects)
        n_to_a = allocation[age_val]
        part_a.extend(subjects[:n_to_a])
        part_b.extend(subjects[n_to_a:])

    assert len(part_a) == n_a, f"Expected {n_a} in A, got {len(part_a)}"
    return part_a, part_b


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def _write_subject_list(path: Path, subject_ids: list[str]) -> None:
    """Write one subject ID per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sid in sorted(subject_ids):
            f.write(sid + "\n")
    log.info("Wrote %d subjects to %s", len(subject_ids), path)


def _build_stats(
    label: str,
    ids: list[str],
    df: pd.DataFrame,
    strategy: str,
    seed: int,
    modality: str,
    total_eligible: int,
) -> dict:
    """Compute age distribution statistics for a partition."""
    sub_df = df[df["sub_id"].isin(ids)]
    age_counts = sub_df["age"].value_counts().sort_index()
    return {
        "label": label,
        "strategy": strategy,
        "seed": seed,
        "modality": modality,
        "n_subjects": len(ids),
        "total_eligible": total_eligible,
        "age_min": float(sub_df["age"].min()),
        "age_max": float(sub_df["age"].max()),
        "age_mean": float(sub_df["age"].mean()),
        "age_median": float(sub_df["age"].median()),
        "age_std": float(sub_df["age"].std()),
        "per_age_counts": {int(k): int(v) for k, v in age_counts.items()},
    }


def _print_summary(
    label: str,
    ids: list[str],
    df: pd.DataFrame,
) -> None:
    """Print age distribution summary for a partition."""
    sub_df = df[df["sub_id"].isin(ids)]
    print(f"\n  {label}: {len(ids)} subjects")
    print(f"    Age range:  {sub_df['age'].min():.0f} – {sub_df['age'].max():.0f}")
    print(f"    Age mean:   {sub_df['age'].mean():.1f}")
    print(f"    Age median: {sub_df['age'].median():.1f}")
    print(f"    Age std:    {sub_df['age'].std():.1f}")
    counts = sub_df["age"].value_counts().sort_index()
    print(f"    Per-age counts: {dict(counts)}")


def _write_stats(path: Path, stats: dict) -> None:
    """Write partition statistics to a plain-text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"label:           {stats['label']}",
        f"strategy:        {stats['strategy']}",
        f"seed:            {stats['seed']}",
        f"modality:        {stats['modality']}",
        f"n_subjects:      {stats['n_subjects']}",
        f"total_eligible:  {stats['total_eligible']}",
        f"age_min:         {stats['age_min']:.0f}",
        f"age_max:         {stats['age_max']:.0f}",
        f"age_mean:        {stats['age_mean']:.2f}",
        f"age_median:      {stats['age_median']:.1f}",
        f"age_std:         {stats['age_std']:.2f}",
        "",
        "per_age_counts:",
    ]
    for age_val, count in stats["per_age_counts"].items():
        lines.append(f"  age {age_val:>3}: {count}")
    path.write_text("\n".join(lines) + "\n")
    log.info("Wrote statistics to %s", path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PNC cohort subject lists.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata", type=Path, required=True,
        help="Path to the metadata CSV (e.g. PNC_ALL_SCORES.csv).",
    )
    parser.add_argument(
        "--sc-dir", type=Path, default=None,
        help="Path to Structural_maps directory.",
    )
    parser.add_argument(
        "--fc-dir", type=Path, default=None,
        help="Path to Functional_Mats directory.",
    )
    parser.add_argument(
        "--modality", type=str, choices=["sc", "fc", "both"], default="sc",
        help="Which subjects to include: sc (SC only), fc (FC only), both (SC and FC).",
    )
    parser.add_argument(
        "--age-col", type=str, default="age_at_cnb",
        help="Name of the age column in the metadata CSV.",
    )
    parser.add_argument(
        "--id-col", type=str, default="SUBJID",
        help="Name of the subject ID column in the metadata CSV.",
    )
    parser.add_argument(
        "--pad-width", type=int, default=10,
        help="Zero-padding width for converting SUBJID to filename IDs.",
    )
    parser.add_argument(
        "--n-young", type=int, default=130,
        help="Number of subjects in partition A.",
    )
    parser.add_argument(
        "--strategy", type=str, choices=["median", "interleaved"], required=True,
        help="Partitioning strategy.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("scripts/cohort_selection/outputs"),
        help="Directory to write output files.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Validate required directories for the chosen modality
    if args.modality in ("sc", "both") and args.sc_dir is None:
        parser.error("--sc-dir is required when --modality is 'sc' or 'both'.")
    if args.modality in ("fc", "both") and args.fc_dir is None:
        parser.error("--fc-dir is required when --modality is 'fc' or 'both'.")

    # Load metadata
    metadata = pd.read_csv(args.metadata, low_memory=False)
    log.info("Loaded metadata: %d rows from %s", len(metadata), args.metadata)

    # Discover available subjects on disk
    sc_ids: set[str] = _discover_sc_subjects(args.sc_dir) if args.sc_dir else set()
    fc_ids: set[str] = _discover_fc_subjects(args.fc_dir) if args.fc_dir else set()
    if args.modality in ("sc", "both"):
        log.info("Found %d subjects with SC data on disk", len(sc_ids))
    if args.modality in ("fc", "both"):
        log.info("Found %d subjects with FC data on disk", len(fc_ids))
    if args.modality == "both":
        log.info("Subjects with both SC and FC: %d", len(sc_ids & fc_ids))

    # Build eligible DataFrame
    eligible = _build_eligible_df(
        metadata, args.id_col, args.age_col, args.pad_width,
        sc_ids, fc_ids, args.modality,
    )
    log.info(
        "Eligible subjects (metadata + %s data on disk + non-null age): %d",
        args.modality, len(eligible),
    )

    # Partition
    if args.strategy == "median":
        part_a, part_b = partition_median(eligible, args.n_young, args.seed)
    elif args.strategy == "interleaved":
        part_a, part_b = partition_interleaved(eligible, args.n_young, args.seed)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    # Validate no overlap
    overlap = set(part_a) & set(part_b)
    if overlap:
        print(f"ERROR: {len(overlap)} subjects in both partitions!", file=sys.stderr)
        sys.exit(1)

    # Summary
    print(f"\nStrategy: {args.strategy}, seed: {args.seed}")
    print(f"Total eligible: {len(eligible)}")
    _print_summary("Partition A (young/sampled)", part_a, eligible)
    _print_summary("Partition B (old/remaining)", part_b, eligible)

    # Write output files
    tag = f"{args.strategy}_s{args.seed}"
    out_a = args.output_dir / f"{tag}_partA_{len(part_a)}.txt"
    out_b = args.output_dir / f"{tag}_partB_{len(part_b)}.txt"
    stats_a = args.output_dir / f"{tag}_partA_{len(part_a)}_stats.txt"
    stats_b = args.output_dir / f"{tag}_partB_{len(part_b)}_stats.txt"

    _write_subject_list(out_a, part_a)
    _write_subject_list(out_b, part_b)
    _write_stats(stats_a, _build_stats(
        "Partition A (young/sampled)", part_a, eligible,
        args.strategy, args.seed, args.modality, len(eligible),
    ))
    _write_stats(stats_b, _build_stats(
        "Partition B (old/remaining)", part_b, eligible,
        args.strategy, args.seed, args.modality, len(eligible),
    ))

    print(f"\nOutput files:")
    print(f"  A subjects: {out_a}")
    print(f"  A stats:    {stats_a}")
    print(f"  B subjects: {out_b}")
    print(f"  B stats:    {stats_b}")


if __name__ == "__main__":
    main()
