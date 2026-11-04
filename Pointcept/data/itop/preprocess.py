import h5py
import numpy as np
import os
import open3d as o3d


BASEDIR = os.path.dirname(os.path.abspath(__file__))


def remove_outliers(
    points: np.ndarray,
    eps: float = 0.05,
    min_points: int = 10,
) -> np.ndarray:
    """
    Keep only the largest DBSCAN cluster from a raw point cloud.

    DBSCAN groups all points into spatial clusters and returns only the biggest
    one, which reliably corresponds to the human body.  Whole disconnected blobs
    that survive segmentation are discarded before normalization, keeping the
    centroid and scale estimate clean.

    Parameters
    ----------
    points     : (N, 3) float array of raw depth points.
    eps        : DBSCAN neighbourhood radius in metres (0.05 m suits ITOP).
    min_points : Minimum cluster size; smaller groups are treated as noise.

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
    """Rotate points 180° around the Y-axis (negate X and Z).
    Aligns ITOP's facing direction with CAPE."""
    flipped = points.copy()
    flipped[:, 0] = -flipped[:, 0]
    flipped[:, 2] = -flipped[:, 2]
    return flipped


# Scenes 4, 6, 7, 8 - evaluation scenes that sum to 4626 frames
EVAL_SCENES = {'04', '06', '07', '08'}


def process_split(pc_file: str, labels_file: str, split_name: str, scene_filter: set = None):
    f = h5py.File(pc_file, 'r')
    labels = h5py.File(labels_file, 'r')

    data = np.asarray(f.get('data'))
    ids = np.asarray(f.get('id'))
    segmentation = np.asarray(labels.get('segmentation'))
    is_valid = np.asarray(labels.get('is_valid'))

    valid_idx = np.where(is_valid == 1)[0]
    data_valid = data[valid_idx]
    seg_valid = segmentation[valid_idx]
    ids_valid = ids[valid_idx]

    saved_ids = []
    saved, skipped, filtered = 0, 0, 0
    for i in range(len(data_valid)):
        frame_id = ids_valid[i].decode()

        # Filter to requested scenes only (scene is the part before '_')
        if scene_filter is not None:
            scene = frame_id.split('_')[0]
            if scene not in scene_filter:
                filtered += 1
                continue

        seg = seg_valid[i].flatten()
        human_pcd = data_valid[i][seg >= 0]

        # Remove outlier clusters before normalization so that stray blobs do
        # not skew the centroid or scale used by normalize_to_half_unit_sphere.
        human_pcd = remove_outliers(human_pcd.astype(np.float64))

        if len(human_pcd) < 2048:
            skipped += 1
            continue

        v_norm = normalize_to_half_unit_sphere(human_pcd)
        pcd_2048 = subsample_fps(v_norm, 2048)
        pcd_2048 = rotate_180_y(pcd_2048)

        np.save(os.path.join(BASEDIR, 'pcds', f'{frame_id}.npy'), pcd_2048)
        saved_ids.append(frame_id)
        saved += 1

        if saved % 100 == 0:
            print(f'[{split_name}] Saved {saved} frames...')

    with open(os.path.join(BASEDIR, 'splits', f'{split_name}.txt'), 'w') as split_file:
        split_file.write('\n'.join(saved_ids))

    print(f'[{split_name}] Done. Saved: {saved}, Skipped (too few points): {skipped}, '
          f'Filtered (wrong scene): {filtered}')
    print(f'[{split_name}] Split file written to splits/{split_name}.txt '
          f'with {len(saved_ids)} entries.')

    f.close()
    labels.close()


os.makedirs(os.path.join(BASEDIR, 'pcds'), exist_ok=True)
os.makedirs(os.path.join(BASEDIR, 'splits'), exist_ok=True)

# The side-view train file is the source of the evaluation scenes: the CAPE-trained
# model is evaluated zero-shot on ITOP, so these frames are only ever a test split.
process_split(
    os.path.join(BASEDIR, 'ITOP_side_train_point_cloud.h5'),
    os.path.join(BASEDIR, 'ITOP_side_train_labels.h5'),
    'test',
    scene_filter=EVAL_SCENES,
)