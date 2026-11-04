"""
Visualize every scene in the ITOP train or test split directly from .h5 files.
Applies the same preprocessing pipeline as preprocess.py:
  outlier removal -> normalize -> FPS subsample -> rotate 180° around Y-axis.

Usage:
  python Pointcept/data/itop/visualize.py                         # train split
  python Pointcept/data/itop/visualize.py --split test
  python Pointcept/data/itop/visualize.py --split train --scene 02
  python Pointcept/data/itop/visualize.py --split test  --delay 50
"""

import argparse
import os
import time
from collections import defaultdict

import h5py
import numpy as np
import open3d as o3d

BASEDIR = os.path.dirname(os.path.abspath(__file__))


# Preprocessing

def remove_outliers(
    points: np.ndarray,
    eps: float = 0.05,
    min_points: int = 10,
) -> np.ndarray:
    """
    Keep only the largest DBSCAN cluster from a raw point cloud.

    Statistical outlier removal only discards isolated stray points; it cannot
    eliminate whole disconnected blobs that survive body segmentation.  DBSCAN
    groups all points into spatial clusters and returns only the biggest one,
    which reliably corresponds to the human body.

    Parameters
    ----------
    points     : (N, 3) float array of raw depth points.
    eps        : DBSCAN neighbourhood radius in metres.  0.05 m works well for
                 ITOP's side-view depth stream; increase slightly if the body
                 is fragmented into multiple clusters.
    min_points : Minimum cluster size; points in smaller groups are treated
                 as noise and discarded.

    Returns
    -------
    Filtered (M, 3) array containing only the dominant body cluster.
    """
    if len(points) == 0:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    labels = np.asarray(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    # labels == -1 means noise; find the cluster with the most points
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) == 0:
        # No cluster found - return original points to avoid dropping the frame
        return points

    largest_label = unique[np.argmax(counts)]
    return points[labels == largest_label]


def normalize_to_half_unit_sphere(vertices: np.ndarray) -> np.ndarray:
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


def rotate_180_y(points: np.ndarray) -> np.ndarray:
    """Rotate 180 degrees around Y-axis (negate X and Z) to align with CAPE."""
    flipped = points.copy()
    flipped[:, 0] = -flipped[:, 0]
    flipped[:, 2] = -flipped[:, 2]
    return flipped


def preprocess(raw_points: np.ndarray, num_points: int = 2048) -> np.ndarray | None:
    """
    Full preprocessing pipeline:
      1. Outlier removal on the raw depth points (largest DBSCAN cluster)
      2. Normalize to half-unit sphere
      3. FPS subsample
      4. Rotate 180° around Y-axis

    Returns None if too few points remain after outlier removal.
    """
    # Step 1 - remove outliers on the raw, un-normalized cloud so that the
    # centroid and scale computed in step 2 are not skewed by stray points.
    cleaned = remove_outliers(raw_points.astype(np.float64))

    if len(cleaned) < num_points:
        return None

    # Step 2 - normalize
    v_norm = normalize_to_half_unit_sphere(cleaned)

    # Step 3 - FPS subsample
    pcd = subsample_fps(v_norm, num_points)

    # Step 4 - rotate
    return rotate_180_y(pcd)


# H5 loading

def load_h5(pc_file: str, labels_file: str) -> list[tuple[str, np.ndarray]]:
    """
    Load all valid frames from an H5 pair.
    Returns a list of (frame_id, raw_points) sorted by frame_id.
    """
    with h5py.File(pc_file, "r") as f, h5py.File(labels_file, "r") as lf:
        data         = np.asarray(f["data"])
        ids          = np.asarray(f["id"])
        segmentation = np.asarray(lf["segmentation"])
        is_valid     = np.asarray(lf["is_valid"])

    frames = []
    for i in np.where(is_valid == 1)[0]:
        seg       = segmentation[i].flatten()
        human_pts = data[i][seg >= 0]
        frame_id  = ids[i].decode()
        frames.append((frame_id, human_pts))

    return sorted(frames, key=lambda x: x[0])


def group_by_scene(
    frames: list[tuple[str, np.ndarray]],
) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Group (frame_id, points) pairs by scene prefix ('02_00001' -> '02')."""
    scenes: dict[str, list] = defaultdict(list)
    for frame_id, pts in frames:
        scene = frame_id.split("_")[0]
        scenes[scene].append((frame_id, pts))
    return dict(sorted(scenes.items()))


# Colorization

def colorize(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Color points by normalized height (Z-axis) with a cool-warm gradient."""
    points = np.asarray(pcd.points)
    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors = np.zeros((len(z_norm), 3))
    colors[:, 0] = z_norm
    colors[:, 1] = 1 - np.abs(z_norm - 0.5) * 2
    colors[:, 2] = 1 - z_norm
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# Scene playback

def play_scene(
    scene_id: str,
    frames: list[tuple[str, np.ndarray]],
    delay_ms: int,
    scene_index: int,
    total_scenes: int,
) -> None:
    """Animate all frames in one scene. Close the window or press Q to advance."""
    print(f"\nScene {scene_id}  [{scene_index}/{total_scenes}]  -  {len(frames)} frame(s)")
    print("  Press Q or Escape to skip to the next scene.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Scene {scene_id}  [{scene_index}/{total_scenes}]  -  {len(frames)} frames",
        width=1024,
        height=768,
    )
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.05])

    # Bootstrap with the first valid preprocessed frame
    pcd = o3d.geometry.PointCloud()
    bootstrapped = False
    for frame_id, raw_pts in frames:
        pts = preprocess(raw_pts)
        if pts is not None:
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd = colorize(pcd)
            bootstrapped = True
            break

    if not bootstrapped:
        print(f"  Scene {scene_id}: all frames skipped (too few points).")
        vis.destroy_window()
        return

    vis.add_geometry(pcd)
    vis.reset_view_point(True)

    for i, (frame_id, raw_pts) in enumerate(frames):
        pts = preprocess(raw_pts)
        if pts is None:
            continue

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(pts)
        new_pcd = colorize(new_pcd)

        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors
        vis.update_geometry(pcd)

        if not vis.poll_events():   # window closed -> advance to next scene
            break
        vis.update_renderer()

        if i % 50 == 0:
            print(f"    Frame {i + 1}/{len(frames)}: {frame_id}")

        time.sleep(delay_ms / 1000.0)

    vis.destroy_window()


# Entry point

H5_FILES = {
    "train": ("ITOP_side_train_point_cloud.h5", "ITOP_side_train_labels.h5"),
    "test":  ("ITOP_side_test_point_cloud.h5",  "ITOP_side_test_labels.h5"),
}


def run(split: str, scene_filter: str | None, delay_ms: int) -> None:
    pc_file, labels_file = (os.path.join(BASEDIR, f) for f in H5_FILES[split])

    print(f"Loading {split} split from H5 files...")
    frames = load_h5(pc_file, labels_file)
    scenes = group_by_scene(frames)

    if scene_filter:
        if scene_filter not in scenes:
            available = ", ".join(scenes.keys())
            raise ValueError(
                f"Scene '{scene_filter}' not in '{split}' split. "
                f"Available: {available}"
            )
        scenes = {scene_filter: scenes[scene_filter]}

    total = len(scenes)
    total_frames = sum(len(v) for v in scenes.values())
    print(f"Split : {split}")
    print(f"Scenes: {total}  ({', '.join(scenes.keys())})")
    print(f"Frames: {total_frames} valid total")

    for idx, (scene_id, scene_frames) in enumerate(scenes.items(), start=1):
        play_scene(
            scene_id=scene_id,
            frames=scene_frames,
            delay_ms=delay_ms,
            scene_index=idx,
            total_scenes=total,
        )

    print("\nAll scenes finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize ITOP scenes directly from .h5 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Which split to visualize.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        metavar="SCENE_ID",
        help="Visualize a single scene only (e.g. --scene 02).",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=100,
        metavar="MS",
        help="Delay between frames in milliseconds.",
    )
    args = parser.parse_args()

    run(split=args.split, scene_filter=args.scene, delay_ms=args.delay)