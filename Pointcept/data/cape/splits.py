"""
Write the train, val and test splits of the CAPE dataset to splits/, using the frames
in pcds/ and the sequence splits in pcds/splits.json produced by preprocess.py.
Each line is the ID of one frame, {subject}_{sequence}.{frame},
e.g. 00032_longshort_hips.000001. Use --dry_run to only print the frame counts.
"""

import argparse
import json
from pathlib import Path


SPLIT_NAMES = ["train", "test", "val"]

# Expected frame counts (from the chosen partition)
TARGETS = {"train": 12_432, "test": 3_311, "val": 2_228}


# Filename parsing

def parse_stem(stem: str) -> tuple[str, str, str]:
    """
    Parse a flat .npy stem into (subject, seq_name, frame_stem).

    Filename format: {subject}_{seq_name}_{frame_stem}
    Subject is always the first token, frame_stem is always the last token,
    seq_name is everything in between.

    Example:
        "00032_longshort_hips_000042"  ->  ("00032", "longshort_hips", "000042")
        "00096_shirtlong_hips_000001"  ->  ("00096", "shirtlong_hips", "000001")
    """
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Cannot parse stem '{stem}': expected at least 3 underscore-separated tokens."
        )
    subject    = parts[0]
    frame_stem = parts[-1]
    seq_name   = "_".join(parts[1:-1])
    return subject, seq_name, frame_stem


# Collection

def collect_frames(input_dir: Path) -> dict[str, list[str]]:
    """
    Walk the flat input_dir, parse each .npy filename, and assign it to
    a split using splits.json.

    Returns a dict: split_name -> sorted list of model IDs.
    """
    json_path = input_dir / "splits.json"
    if not json_path.exists():
        raise SystemExit(
            f"splits.json not found in {input_dir}.\n"
            "Run preprocess.py first to generate it."
        )

    with open(json_path) as f:
        splits_json: dict[str, list[str]] = json.load(f)

    # Build reverse lookup:  "subject/seq_name"  ->  split_name
    seq_to_split: dict[str, str] = {}
    for split_name, sequences in splits_json.items():
        for seq_key in sequences:           # e.g. "00032/longshort_hips"
            seq_to_split[seq_key] = split_name

    result: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}
    unmatched: list[str] = []

    for npy_file in sorted(input_dir.glob("*.npy")):
        try:
            subject, seq_name, frame_stem = parse_stem(npy_file.stem)
        except ValueError as e:
            print(f"  [WARN] Skipping {npy_file.name}: {e}")
            continue

        seq_key  = f"{subject}/{seq_name}"
        split    = seq_to_split.get(seq_key)

        if split is None:
            unmatched.append(npy_file.name)
            continue

        model_id = f"{subject}_{seq_name}.{frame_stem}"
        result[split].append(model_id)

    if unmatched:
        print(f"\n  [WARN] {len(unmatched)} file(s) not found in splits.json and were skipped:")
        for name in unmatched[:10]:
            print(f"    {name}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more.")

    return result


# Main

def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate frame-level train/test/val txt files from a flat pcds/ folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", type=Path,
        default=script_dir / "pcds",
        help="Flat pcds/ directory containing .npy files and splits.json.",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=script_dir / "splits",
        help="Directory where train.txt, test.txt, val.txt will be written.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print summary without writing any files.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    split_ids = collect_frames(args.input_dir)

    # Summary table
    total = sum(len(v) for v in split_ids.values())
    print(f"\nInput:  {args.input_dir.resolve()}")
    print(f"Output: {args.output_dir.resolve()}\n")
    print(f"{'Split':<8} {'Sequences':>10} {'Frames':>10}  {'Target':>10}  {'Diff':>8}")
    print("-" * 56)

    for split in SPLIT_NAMES:
        ids      = split_ids[split]
        prefixes = {mid.rsplit(".", 1)[0] for mid in ids}   # unique subject_seq keys
        diff     = len(ids) - TARGETS[split]
        print(f"{split:<8} {len(prefixes):>10} {len(ids):>10,}  {TARGETS[split]:>10,}  {diff:>+8}")

    print(f"{'total':<8} {'':>10} {total:>10,}")
    print()

    if args.dry_run:
        print("Dry run - no files written.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        out_path = args.output_dir / f"{split}.txt"
        out_path.write_text("\n".join(split_ids[split]) + "\n")
        print(f"  Written: {out_path}  ({len(split_ids[split]):,} entries)")

    print(f"\nDone. Split files saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()