"""
Preprocess the 00032 and 00096 sequences of the CAPE dataset into point clouds.
The .npz frames are read from data/, laid out as in Pointcept/data/README.md, and each
frame is saved to pcds/ as {subject}_{sequence}_{frame}.npy, with the split of each
sequence in pcds/splits.json. See --help for the options.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# Core conversion

def normalize_to_half_unit_sphere(vertices: np.ndarray) -> np.ndarray:
    """
    Center vertices and scale so that the maximum point norm equals 0.5.
    This matches the observed range of the reference .npy files (~[-0.49, 0.49]).
    """
    centroid = vertices.mean(axis=0)
    v_centered = vertices - centroid
    max_norm = np.max(np.linalg.norm(v_centered, axis=1))
    if max_norm < 1e-8:
        raise ValueError("All vertices are at the same location - degenerate mesh.")
    return v_centered / max_norm * 0.5


def subsample_fps(vertices: np.ndarray, num_points: int) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd = pcd.farthest_point_down_sample(num_points)
    return np.asarray(pcd.points)


def npz_to_npy(
    npz_path: Path,
    output_path: Path,
    num_points: int = 2048,
    rng: np.random.Generator = None,
    key: str = "v_posed",
    use_fps: bool = False,
) -> None:
    """
    Load one .npz, extract canonical vertices, normalize, subsample, save .npy.

    Args:
        npz_path:    Path to the source .npz file.
        output_path: Destination .npy path (parent dirs are created automatically).
        num_points:  Number of points to sample (default: 2048).
        rng:         NumPy random Generator for reproducibility (ignored with use_fps).
        key:         Array key to extract from the .npz (default: 'v_posed').
        use_fps:     Use farthest point sampling for spatially uniform coverage.
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.load(npz_path, allow_pickle=True)

    if key not in data:
        available = list(data.keys())
        raise KeyError(
            f"Key '{key}' not found in {npz_path.name}. "
            f"Available keys: {available}"
        )

    vertices = data[key]  # (6890, 3)

    # 1. Normalize
    v_norm = normalize_to_half_unit_sphere(vertices)

    # 2. Subsample
    if use_fps:
        point_cloud = subsample_fps(v_norm, num_points)
    else:
        n_available = len(v_norm)
        if n_available < num_points:
            idx = rng.choice(n_available, size=num_points, replace=True)
        else:
            idx = rng.choice(n_available, size=num_points, replace=False)
        point_cloud = v_norm[idx]

    # 3. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, point_cloud)


# Hardcoded splits (subject / sequence_name)
SPLITS = {
    "train": [
        # 00032 (9 sequences)
        ("00032", "longshort_hips"),
        ("00032", "shortlong_ATUsquat"),
        ("00032", "shortshort_chicken_wings"),
        ("00032", "shortshort_hips"),
        ("00032", "shortshort_pose_model"),
        ("00032", "shortshort_punching"),
        ("00032", "shortshort_shoulders_mill"),
        ("00032", "shortshort_ATUsquat"),
        ("00032", "shortshort_soccer"),
        # 00096 (31 sequences)
        ("00096", "shirtlong_bend_back_and_front"),
        ("00096", "shirtlong_hips"),
        ("00096", "shirtlong_punching"),
        ("00096", "shirtlong_shoulders_mill"),
        ("00096", "shirtlong_ATUsquat"),
        ("00096", "shirtlong_soccer"),
        ("00096", "shirtlong_squats"),
        ("00096", "shirtlong_tilt_twist_left"),
        ("00096", "shirtshort_ballerina_spin"),
        ("00096", "shirtshort_flying_eagle"),
        ("00096", "shirtshort_hips"),
        ("00096", "shirtshort_pose_model"),
        ("00096", "shirtshort_punching"),
        ("00096", "shirtshort_shoulders_mill"),
        ("00096", "shirtshort_soccer"),
        ("00096", "shirtshort_squats"),
        ("00096", "shirtshort_tilt_twist_left"),
        ("00096", "jerseyshort_bend_back_and_front"),
        ("00096", "jerseyshort_flying_eagle"),
        ("00096", "jerseyshort_hips"),
        ("00096", "jerseyshort_pose_model"),
        ("00096", "jerseyshort_punching"),
        ("00096", "jerseyshort_shoulders_mill"),
        ("00096", "jerseyshort_soccer"),
        ("00096", "jerseyshort_squats"),
        ("00096", "jerseyshort_tilt_twist_left"),
        ("00096", "longshort_ballerina_spin"),
        ("00096", "longshort_bend_back_and_front"),
        ("00096", "longshort_hips"),
        ("00096", "longshort_pose_model"),
        ("00096", "longshort_punching"),
    ],
    "test": [
        # 00032 (11 sequences)
        ("00032", "longshort_ATUsquat"),
        ("00032", "longshort_tilt_twist_left"),
        ("00032", "shortlong_ballerina_spin"),
        ("00032", "shortlong_bend_back_and_front"),
        ("00032", "shortlong_flying_eagle"),
        ("00032", "shortlong_shoulders_mill"),
        ("00032", "shortlong_soccer"),
        ("00032", "shortlong_squats"),
        ("00032", "shortlong_tilt_twist_left"),
        ("00032", "shortshort_bend_back_and_front"),
        ("00032", "shortshort_flying_eagle"),
    ],
    "val": [
        # 00032 (10 sequences)
        ("00032", "longshort_flying_eagle"),
        ("00032", "longshort_pose_model"),
        ("00032", "longshort_punching"),
        ("00032", "longshort_shoulders_mill"),
        ("00032", "longshort_soccer"),
        ("00032", "longshort_squats"),
        ("00032", "shortlong_basketball"),
        ("00032", "shortlong_hips"),
        ("00032", "shortlong_pose_model"),
        ("00032", "shortlong_punching"),
    ],
}


# Processing

def process_splits(
    data_dir: Path,
    output_dir: Path,
    num_points: int,
    seed: "int | None",
    key: str,
    use_fps: bool,
) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_ok = total_fail = total_missing = 0

    for split_name, sequences in SPLITS.items():
        split_ok = split_fail = split_missing = 0
        print(f"\n{'='*60}")
        print(f"  Split: {split_name.upper()}  ({len(sequences)} sequences)")
        print(f"{'='*60}")

        for subject, seq_name in sequences:
            seq_dir = data_dir / subject / seq_name
            if not seq_dir.exists():
                print(f"  [MISSING]  {subject}/{seq_name}  - folder not found: {seq_dir}")
                split_missing += 1
                continue

            npz_files = sorted(seq_dir.glob("*.npz"))
            if not npz_files:
                print(f"  [EMPTY]    {subject}/{seq_name}  - no .npz files found")
                split_missing += 1
                continue

            seq_ok = seq_fail = 0

            for npz_path in npz_files:
                # Flat filename: {subject}_{seq_name}_{original_stem}.npy
                frame_id  = npz_path.stem[len(seq_name):].lstrip("_.")
                flat_name = f"{subject}_{seq_name}_{frame_id}.npy"
                npy_path = output_dir / flat_name
                try:
                    npz_to_npy(
                        npz_path, npy_path,
                        num_points=num_points, rng=rng, key=key, use_fps=use_fps,
                    )
                    seq_ok += 1
                except Exception as e:
                    print(f"    [FAIL]  {npz_path.name}  -  {e}")
                    seq_fail += 1

            status = "OK" if seq_fail == 0 else "!!"
            print(
                f"  [{status}]  {subject}/{seq_name:<40}  "
                f"{seq_ok} frames converted"
                + (f", {seq_fail} failed" if seq_fail else "")
            )
            split_ok   += seq_ok
            split_fail += seq_fail

        print(f"\n  {split_name.upper()} summary: {split_ok} frames OK, "
              f"{split_fail} failed, {split_missing} sequences missing.")
        total_ok      += split_ok
        total_fail    += split_fail
        total_missing += split_missing

    # Save splits.json
    splits_json = {
        split: [f"{subj}/{seq}" for subj, seq in seqs]
        for split, seqs in SPLITS.items()
    }
    json_path = output_dir / "splits.json"
    with open(json_path, "w") as f:
        json.dump(splits_json, f, indent=2)
    print(f"\n  Split assignments saved to: {json_path.resolve()}")

    print(f"\n{'='*60}")
    print(f"  ALL DONE: {total_ok} frames converted, "
          f"{total_fail} failed, {total_missing} sequences missing.")
    print(f"  Output root: {output_dir.resolve()}")
    print(f"{'='*60}")


# CLI

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Preprocess CAPE .npz files into a flat pcds/ folder with splits.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=Path,
        default=script_dir / "data",
        help="Root folder containing 00032/ and 00096/ subject directories.",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=script_dir / "pcds",
        help="Output folder. All .npy files are written flat here alongside splits.json.",
    )
    parser.add_argument("--num_points", type=int, default=2048,
                        help="Number of points to sample per frame.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible subsampling.")
    parser.add_argument("--key", type=str, default="v_posed",
                        help="Array key to extract from each .npz.")
    parser.add_argument("--no_fps", action="store_true",
                        help="Use random sampling instead of farthest point sampling.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.data_dir.exists():
        sys.exit(f"ERROR: data directory not found: {args.data_dir}")

    process_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_points=args.num_points,
        seed=args.seed,
        key=args.key,
        use_fps=not args.no_fps,
    )