import os
import sys
import numpy as np
import open3d as o3d

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from method.graphs.knngraph import KnnGraph
from method.detectors.hmaximadetector import HMaximaDetector
from method.detectors.peakdetector import PeakDetector
from method.detectors.fpsdetector import FPSDetector
from method.detectors.issdetector import ISSDetector
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances_argmin_min


class HumanGEKey:
    def __init__(
            self,
            split='all',
            data_root="Pointcept/data/smpl/pcds",
            knn=20,
            algorithm='hmaxima',
            num_keys=10,
            visualization=False,
    ):
        self.split = split
        self.data_root = data_root
        self.algorithm = algorithm
        self.knn = knn
        self.num_keys = num_keys
        self.visualization = visualization
        self.categories = []
        self.token2category = {}

        ious = 0
        dists = 0
        num_exp = 0
        desired_keypoint_num = self.num_keys

        start = 0
        num_pairs = 5

        for pair_id in range(num_pairs):
            template_idx = None
            base_idx = start + pair_id * 2  # 50, 52, 54, ...

            for i in range(base_idx, base_idx + 2):  # (50, 51), (52, 53), ...
                person = os.path.join(project_root, self.data_root, f"pose_{i:04d}.pcd")
                pcd = o3d.io.read_point_cloud(person)
                coord = np.asarray(pcd.points)

                knn_graph = KnnGraph(coord, k=self.knn, use_fps=True, n_farthest=100)
                G, geo_dists = knn_graph.get_graph_and_distances()

                if algorithm == 'peak':
                    detector = PeakDetector(G, geo_dists, coord, num_peaks=self.num_keys)
                    maxima = detector.get_peak_spheres(radius=0.03)
                elif algorithm == 'hmaxima':
                    detector = HMaximaDetector(G, geo_dists, coord, target_maxima=self.num_keys)
                    maxima = detector.get_hmaxima_spheres()
                elif algorithm == 'fps':
                    detector = FPSDetector(coord, num_samples=self.num_keys)
                    maxima = detector.get_fps_spheres()
                elif algorithm == 'iss':
                    detector = ISSDetector(pcd, coord, num_keys=self.num_keys)
                    maxima = detector.get_iss_spheres()
                else:
                    raise ValueError(f"Unsupported algorithm: '{algorithm}', use 'hmaxima', 'peak', or 'fps'")

                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(coord)
                if self.visualization:
                    use_spheres = True
                    geometries = maxima
                    if use_spheres:
                        sphere_radius = 0.004
                        grey_spheres = []
                        for point in coord:
                            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                            sphere.translate(point)
                            sphere.paint_uniform_color([0.5, 0.5, 0.5])
                            sphere.compute_vertex_normals()
                            grey_spheres.append(sphere)
                        geometries = grey_spheres + geometries
                    else:
                        geometries = [cloud] + geometries

                    o3d.visualization.draw_geometries(geometries)

                pc = coord
                kp = np.array([kp_sphere.get_center() for kp_sphere in maxima])

                if i == base_idx:
                    template_idx = pairwise_distances_argmin(kp, pc)
                    labels = np.ones((pc.shape[0]), int)
                    labels[np.array(template_idx)] = 2
                    template_idx = set(template_idx)
                else:
                    template_idx2 = pairwise_distances_argmin(kp, pc)
                    transferred_points = pc[list(template_idx)]
                    _, dist = pairwise_distances_argmin_min(transferred_points, pc[template_idx2])

                    template_idx2 = set(template_idx2)
                    iou = len(template_idx & template_idx2) / (len(template_idx | template_idx2))
                    ious += iou
                    dists += np.mean(dist)
                    num_exp += 1

        print('{} keypoints: iou {:.4f}, dist {:.4f}'.format(desired_keypoint_num, ious / num_exp, dists / num_exp))


if __name__ == "__main__":
    HumanGEKey(knn=20, algorithm='peak', num_keys=10, visualization=False)
