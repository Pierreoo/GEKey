import os
import numpy as np
import open3d as o3d


def load_scanobjectnn(bin_path):
    data = np.fromfile(bin_path, dtype=np.float32)
    num_points = int(data[0])
    points = data[1:].reshape(num_points, 11)

    xyz      = points[:, 0:3]
    normals  = points[:, 3:6]
    rgb      = points[:, 6:9]
    label    = points[:, 9]
    instance = points[:, 10]

    return xyz, normals, rgb, label, instance


input_dir  = "chair"
output_dir = "pcds"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith(".bin") or "_indices" in fname or "_part" in fname:
        continue

    bin_path = os.path.join(input_dir, fname)
    xyz, normals, rgb, label, instance = load_scanobjectnn(bin_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors  = o3d.utility.Vector3dVector(rgb / 255.0)

    out_path = os.path.join(output_dir, fname.replace(".bin", ".pcd"))
    if not os.path.exists(out_path):
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"Saved: {out_path}")
    else:
        print(f"Skipped (already exists): {out_path}")

print("Done!")