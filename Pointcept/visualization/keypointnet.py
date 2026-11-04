"""
Visualize the predicted keypoints on the point cloud of the KeyPointNet dataset
"""

import os
import json
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


def get_keypoints(annotation_file, model_id):
    with open(annotation_file, 'r') as f:
        annots = json.load(f)
    keypoints_dict = {
        annot['model_id']: [
            (kp_info['pcd_info']['point_index'], kp_info['semantic_id'])
            for kp_info in annot['keypoints']
        ]
        for annot in annots
    }
    return keypoints_dict.get(model_id, [])

# Airplane: pred_key_path = "exp/shapenet/airplane/result/02691156-99c13d3c3f30c34dca8607f540cc62ba.npy"
# Bed: pred_key_path = "exp/shapenet/bed/result/02818832-71ebb1db4d8ef7c51db05bb2d8c217b.npy"
# Bottle: pred_key_path = "exp/shapenet/bottle/result/02876657-412d5f41b5fda0b271dbcf1c4061c69b.npy"
# Cap: pred_key_path = "exp/shapenet/cap/result/02954340-9c225cd8f50b7c353d9199581d0f4b4.npy"
# Car: pred_key_path = "exp/shapenet/car/result/02958343-87ee241d3d0d1dda4ff3c6764341833.npy"
# Chair: pred_key_path = "exp/shapenet/chair/result/03001627-a5eea0cf3865da9749fb53621b8edde7.npy"
# Guitar: pred_key_path = "exp/shapenet/guitar/result/03467517-22b21e858c58a3d59a6208793b9dfca.npy"
# Helmet: pred_key_path = "exp/shapenet/helmet/result/03513137-3621cf047be0d1ae52fafb0cab311e6a.npy"
# Knife: pred_key_path = "exp/shapenet/knife/result/03624134-a2288d5f3a44233bc40c6b891c4913bd.npy"
# Motorcycle: pred_key_path = "exp/shapenet/motorcycle/result/03790512-33aa1eadcd21734148272812609617e2.npy"
# Mug: pred_key_path = "exp/shapenet/mug/result/03797390-46ed9dad0440c043d33646b0990bb4a.npy"
# Table: pred_key_path = "exp/shapenet/table/result/04379243-a59f0a9956137c0735962325b2299f4c.npy"
# Vessel: pred_key_path = "exp/shapenet/vessel/result/04530566-739a6f209a49040a4f44d84b9baa5f42.npy"

pred_key_path = "../exp/shapenet/airplane/result/02691156-7a80f3950b5e31f6b0d51feef5761dce.npy"
filename = os.path.splitext(os.path.basename(pred_key_path))[0]
category_id, model_id = filename.split('-')
data_path = f"../exp/shapenet/airplane/pcds/{category_id}-{model_id}.pcd"
annotation_path = '../data/keypointnet/annotations/airplane.json'

pcd = load_point_cloud(data_path, uniform_color=[0.5, 0.5, 0.5])
coord = np.asarray(pcd.points)
sphere_radius = 0.0025
grey_spheres = []
for point in coord:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere.translate(point)
    sphere.paint_uniform_color([0.5, 0.5, 0.5])
    sphere.compute_vertex_normals()
    grey_spheres.append(sphere)

keypoints_gt = get_keypoints(annotation_path, model_id)
gt_points = np.array([pcd.points[idx] for idx, _ in keypoints_gt])
gt_colors = np.array([[0, 0, 1]] * len(gt_points))
gt_sphere_pcds = create_colored_shapes(gt_points, gt_colors, radius=0.005, shape="sphere")

keypoints_pred = np.load(pred_key_path)
pred_points = keypoints_pred

alignment = True
if alignment:
    cmap = sns.color_palette("deep", 10)
    pred_colors = np.array(cmap)
else:
    pred_colors = np.array([[0, 0.4, 0.1]] * 10)

assert keypoints_pred.shape[0] == 10, "Expected 10 predicted keypoints."
pred_sphere_pcds = create_colored_shapes(keypoints_pred, pred_colors, radius=0.023, shape="sphere")

o3d.visualization.draw_geometries(
    grey_spheres + pred_sphere_pcds + gt_sphere_pcds,
    window_name="Keypoints: GT (blue) & Predicted (colored spheres)"
)
