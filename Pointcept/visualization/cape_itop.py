"""
Visualize CAPE and ITOP point clouds with predicted keypoints at their absolute positions.
Both are rendered without any offset to check centering around the origin.
"""

import os
import open3d as o3d
import numpy as np
import seaborn as sns

# Color palette (shared)
deep10 = sns.color_palette("deep", 10)
extra = [
    (0.35, 0.35, 0.35),
    (0.89, 0.10, 0.11),
]
KP_COLORS = deep10 + extra


def create_sphere_cloud(points, color, radius=0.004):
    spheres = []
    for point in points:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.translate(point)
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        spheres.append(mesh)
    return spheres


def create_keypoint_spheres(keypoints, colors, radius=0.03):
    spheres = []
    for point, color in zip(keypoints, colors):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.translate(point)
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        spheres.append(mesh)
    return spheres


def load_pcd_and_keypoints(data_path, pred_key_path, pcd_color):
    pcd = o3d.io.read_point_cloud(data_path)
    coords = np.asarray(pcd.points)
    print(f"  PCD centroid : {coords.mean(axis=0).round(4)}")
    print(f"  PCD min      : {coords.min(axis=0).round(4)}")
    print(f"  PCD max      : {coords.max(axis=0).round(4)}")

    keypoints = np.load(pred_key_path)
    assert keypoints.shape[0] == 12, \
        f"Expected 12 keypoints, got {keypoints.shape[0]} in {pred_key_path}"

    spheres   = create_sphere_cloud(coords, pcd_color)
    kp_meshes = create_keypoint_spheres(keypoints, KP_COLORS)
    return spheres + kp_meshes


# CAPE
cape_subject  = "00032"
cape_sequence = "shortlong_soccer"
cape_frame    = "000151"
cape_exp_dir  = "../exp/cape/person"

cape_name     = f"{cape_subject}_{cape_sequence}_{cape_frame}"
cape_pcd_path = os.path.join(cape_exp_dir, "pcds",   f"{cape_name}.pcd")
cape_kp_path  = os.path.join(cape_exp_dir, "result", f"{cape_name}.npy")

# ITOP
itop_scene   = "08"
itop_frame   = "00449"
itop_exp_dir = "../exp/itop/person"

itop_name     = f"{itop_scene}_{itop_frame}"
itop_pcd_path = os.path.join(itop_exp_dir, "pcds",   f"{itop_name}.pcd")
itop_kp_path  = os.path.join(itop_exp_dir, "result", f"{itop_name}.npy")

# Build geometry lists
print(f"\nCAPE: {cape_name}")
cape_geoms = load_pcd_and_keypoints(cape_pcd_path, cape_kp_path, pcd_color=[0.6, 0.8, 1.0])

print(f"\nITOP: {itop_name}")
itop_geoms = load_pcd_and_keypoints(itop_pcd_path, itop_kp_path, pcd_color=[1.0, 0.8, 0.6])

# Origin marker so you can see where 0,0,0 is
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

print("\nBlue = CAPE  |  Orange = ITOP  |  Coordinate frame = origin (0,0,0)")

o3d.visualization.draw_geometries(
    cape_geoms + itop_geoms + [origin],
    window_name=f"Absolute positions - CAPE ({cape_name})  vs  ITOP ({itop_name})",
    width=1400,
    height=800,
)