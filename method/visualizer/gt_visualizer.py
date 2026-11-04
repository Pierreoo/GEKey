import json
import open3d as o3d
import numpy as np


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


class GTKeypointVisualizer:
    def __init__(self, annot_file, model_id, cloud):
        self.annot_file = annot_file
        self.model_id = model_id
        self.cloud = cloud

    def get_gt_spheres(self):
        keypoints = get_keypoints(self.annot_file, self.model_id)

        gt_sphere_pcds = []
        for idx, cls in keypoints:
            keypoint_position = np.asarray(self.cloud.points)[idx]
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(keypoint_position)
            sphere.paint_uniform_color([0.0, 0.0, 1.0])
            gt_sphere_pcds.append(sphere)

        return gt_sphere_pcds
