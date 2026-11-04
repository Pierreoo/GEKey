"""
Visualize the predicted keypoints on the chair point cloud of the ScanObjectNN dataset
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


def fps(coord, num_points):
    idx = np.zeros(num_points, dtype=int)
    idx[0] = np.random.randint(len(coord))
    distances = np.full(len(coord), np.inf)

    for i in range(1, num_points):
        last = coord[idx[i - 1]]
        dist = np.sum((coord - last) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        idx[i] = np.argmax(distances)

    return idx


def sample_points(coord, num_points=2048, method="random"):
    if method == "random":
        idx = np.random.choice(len(coord), num_points, replace=False)
    elif method == "fps":
        idx = fps(coord, num_points)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    return idx

# Experiment folder and a chair from ../data/scanobjectnn/splits/chair.txt
exp_dir = "../exp/scanobjectnn/<exp_name>"
filename = "scene0038_00_00013"
pred_key_path = os.path.join(exp_dir, "result", f"{filename}.npy")
data_path = os.path.join(exp_dir, "pcds", f"{filename}.pcd")

pcd = load_point_cloud(data_path, uniform_color=[0.5, 0.5, 0.5])
coord = np.asarray(pcd.points)
idx = sample_points(coord, num_points=2048, method="random")
coord = coord[idx]
sphere_radius = 0.012
grey_spheres = []
for point in coord:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere.translate(point)
    sphere.paint_uniform_color([0.5, 0.5, 0.5])
    sphere.compute_vertex_normals()
    grey_spheres.append(sphere)

keypoints_pred = np.load(pred_key_path)
pred_points = keypoints_pred

alignment = True
if alignment:
    cmap = sns.color_palette("deep", 10)
    pred_colors = np.array(cmap)
else:
    pred_colors = np.array([[0, 0.4, 0.1]] * 10)

assert keypoints_pred.shape[0] == 10, "Expected 10 predicted keypoints."
pred_sphere_pcds = create_colored_shapes(keypoints_pred, pred_colors, radius=0.06, shape="sphere")

o3d.visualization.draw_geometries(
    grey_spheres + pred_sphere_pcds,
    window_name="Predicted (colored spheres)"
)
