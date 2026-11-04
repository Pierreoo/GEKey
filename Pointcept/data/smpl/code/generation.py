"""
Generates 1000 randomly posed SMPL bodies as point clouds in ../pcds/.
Requires the SMPL model and code, which are not part of this repository, see ../README.md.
Pose and shape sampling follows hello_smpl.py from the SMPL code.
"""
import os
import numpy as np
import open3d as o3d
from serialization import load_model


base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(base_dir, '..', 'models', 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'))

m = load_model(model_path)


output_dir = os.path.normpath(os.path.join(base_dir, '..', 'pcds'))
os.makedirs(output_dir, exist_ok=True)

for i in range(1000):
    m.pose[:] = np.random.rand(m.pose.size) * .2
    m.betas[:] = np.random.rand(m.betas.size) * .03

    vertices = np.array(m.r)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_point_cloud(os.path.join(output_dir, f'pose_{i:04d}.pcd'), pcd)

    if i % 100 == 0:
        print(f'Saved {i}/1000 poses...')

print('Done! 1000 poses saved to', output_dir)