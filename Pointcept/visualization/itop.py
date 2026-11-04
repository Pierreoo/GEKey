"""
Visualize the predicted keypoints on the person point cloud of the ITOP dataset.

Names are stored as {scene}_{frame}, e.g. 00_00053
"""

import os
import open3d as o3d
import numpy as np
import seaborn as sns


def load_point_cloud(file_path, uniform_color=None):
    pcd = o3d.io.read_point_cloud(file_path)
    if uniform_color:
        pcd.paint_uniform_color(uniform_color)
    return pcd


def create_colored_shapes(points, colors, radius=0.01, shape="sphere"):
    shapes = []
    for point, color in zip(points, colors):
        if shape == "sphere":
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        elif shape == "tetrahedron":
            mesh = o3d.geometry.TriangleMesh.create_tetrahedron(radius=radius)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
        mesh.translate(point)
        mesh.paint_uniform_color(color)
        shapes.append(mesh)
    return shapes


scene = "08"
frame = "02194"

exp_dir = "../exp/itop/person"

safe_name     = f"{scene}_{frame}"
pred_key_path = os.path.join(exp_dir, "result", f"{safe_name}.npy")
data_path     = os.path.join(exp_dir, "pcds",   f"{safe_name}.pcd")


# Load point cloud
pcd = load_point_cloud(data_path, uniform_color=[0.5, 0.5, 0.5])
coord = np.asarray(pcd.points)

sphere_radius = 0.004
grey_spheres = []
for point in coord:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere.translate(point)
    sphere.paint_uniform_color([0.5, 0.5, 0.5])
    grey_spheres.append(sphere)

keypoints_pred = np.load(pred_key_path)
assert keypoints_pred.shape[0] == 12, f"Expected 12 predicted keypoints, got {keypoints_pred.shape[0]}."

alignment = True
if alignment:
    deep10 = sns.color_palette("deep", 10)
    extra = [
        (0.35, 0.35, 0.35),  # dark gray
        (0.89, 0.10, 0.11),  # strong red
    ]
    cmap = deep10 + extra
    pred_colors = np.array(cmap)
else:
    pred_colors = np.array([[0, 0.4, 0.1]] * 12)

pred_sphere_pcds = create_colored_shapes(keypoints_pred, pred_colors, radius=0.03, shape="sphere")

o3d.visualization.draw_geometries(
    grey_spheres + pred_sphere_pcds,
    window_name=f"ITOP - {safe_name} - Predicted keypoints",
)