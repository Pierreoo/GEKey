"""
Plot and visualize the predicted eccentricity values against the ground truth
"""


import os
import open3d as o3d
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


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


def point_cloud_to_spheres(pcd, radius=0.005):
    points = np.asarray(pcd.points)
    color = pcd.colors[0] if pcd.has_colors() else [0.5, 0.5, 0.5]

    sphere_meshes = []
    for pt in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        sphere_meshes.append(sphere)

    return sphere_meshes


pred_key_path = "exp/keypointnet/airplanege/result/02691156-73945c1b294716e1d041917556492646_ecc_pred.npy"
gt_key_path = "exp/keypointnet/airplanege/result/02691156-73945c1b294716e1d041917556492646_ecc_gt.npy"
filename = os.path.splitext(os.path.basename(pred_key_path))[0]
category_id, model_id = filename.split('-')
data_path = f"../data/keypointnet/pcds/{category_id}/73945c1b294716e1d041917556492646.pcd"

pred_values = np.load(pred_key_path)
gt_values = np.load(gt_key_path)

plt.figure(figsize=(8, 5))

plt.hist(pred_values, bins=50, alpha=0.6, label="Prediction", density=True)
plt.hist(gt_values, bins=50, alpha=0.6, label="Ground Truth", density=True)

plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Histogram of Prediction vs Ground Truth")
plt.legend()
plt.tight_layout()
plt.show()

pred_norm = (pred_values - pred_values.min()) / (pred_values.ptp() + 1e-8)
gt_norm = (gt_values - gt_values.min()) / (gt_values.ptp() + 1e-8)


pred_colors = cm.plasma(pred_values)[:, :3]  # RGB only
gt_colors = cm.plasma(gt_values)[:, :3]

pcd = o3d.io.read_point_cloud(data_path)
points = np.asarray(pcd.points)

def create_colored_spheres(points, colors, radius=0.002):
    spheres = []
    for pt, color in zip(points, colors):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
        mesh.translate(pt)
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        spheres.append(mesh)
    return spheres

pred_spheres = create_colored_spheres(points, pred_colors, radius=0.01)
gt_spheres = create_colored_spheres(points, gt_colors, radius=0.01)

o3d.visualization.draw_geometries(pred_spheres, window_name="Predicted Colors")
o3d.visualization.draw_geometries(gt_spheres, window_name="Ground Truth Colors")


differences = np.abs(pred_values - gt_values)
diff_norm = (differences - differences.min()) / (differences.ptp() + 1e-8)
diff_colors = cm.plasma(differences)[:, :3]
diff_spheres = create_colored_spheres(points, diff_colors, radius=0.01)
o3d.visualization.draw_geometries(diff_spheres, window_name="Difference Colors")
