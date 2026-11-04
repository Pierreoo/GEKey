import os
import sys
import torch
import numpy as np
import open3d as o3d

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from method.graphs.knngraph import KnnGraph
from method.graphs.delaunay import DelaunayGraph
from method.detectors.hmaximadetector import HMaximaDetector
from method.detectors.peakdetector import PeakDetector
from method.detectors.fpsdetector import FPSDetector
from method.metrics import coverage, inclusivity


class KeypointNetGEKey:
    def __init__(
            self,
            data_root="Pointcept/data/scanobjectnn",
            root_cloud="Pointcept/data/scanobjectnn/pcds",
            category="chair",
            graph='knn',
            knn=20,
            algorithm='hmaxima',
            num_keys=10,
            visualization=False,
    ):
        self.data_root = data_root
        self.root_cloud = root_cloud
        self.category = category.lower()
        self.graph = graph
        self.algorithm = algorithm
        self.knn = knn
        self.num_keys = num_keys
        self.visualization = visualization
        self.categories = []
        self.token2category = {}

        splits_path = os.path.join(self.data_root, 'splits')
        split_path = os.path.join(project_root, splits_path, "{}.txt".format(self.category))
        data_list = np.loadtxt(split_path, dtype="str")

        pred_all = []
        coverage_all = []
        inclusivity_all = []
        for idx, data in enumerate(data_list):
            data_idx = idx % len(data_list)
            data_name = data_list[data_idx]

            cloud_path = os.path.join(project_root, self.root_cloud)
            cloud_file = os.path.join(project_root, cloud_path, f"{data_name}.pcd")

            cloud = o3d.io.read_point_cloud(cloud_file)
            cloud.paint_uniform_color([0.5, 0.5, 0.5])
            coord = np.asarray(cloud.points)
            idx = sample_points(coord, num_points=2048, method="random")
            coord = coord[idx]

            if self.graph == 'knn':
                graph = KnnGraph(coord, k=self.knn, use_fps=True, n_farthest=100)
            elif self.graph == 'delaunay':
                graph = DelaunayGraph(coord)
            G, geo_dists = graph.get_graph_and_distances()

            if self.visualization:
                graph.visualize()

            if algorithm == 'peak':
                detector = PeakDetector(G, geo_dists, coord, num_peaks=self.num_keys)
                maxima = detector.get_peak_spheres()
                if visualization:
                    heights = detector.compute_heights()
                    detector.visualize_heights(heights)
            elif algorithm == 'hmaxima':
                detector = HMaximaDetector(G, geo_dists, coord, target_maxima=self.num_keys)
                maxima = detector.get_hmaxima_spheres()
            elif algorithm =='fps':
                detector = FPSDetector(coord, num_samples=self.num_keys)
                maxima = detector.get_fps_spheres()
            else:
                print(f"Unsupported algorithm: '{algorithm}', use 'hmaxima', 'peak' or 'fps'")

            if self.visualization:
                use_spheres = True
                geometries = maxima
                if use_spheres:
                    sphere_radius = 0.0015
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

            centers = np.array([s.get_center() for s in maxima])
            pred_keypoints = torch.from_numpy(centers).float()
            coord = torch.from_numpy(coord).float()

            inclusivity_score = inclusivity(pred_keypoints, coord, threshold=0.1)
            coverage_score = coverage(pred_keypoints.unsqueeze(0), coord.unsqueeze(0))

            inclusivity_all.append(torch.mean(inclusivity_score))
            coverage_all.append(coverage_score)

            pred_all.append(pred_keypoints)


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


if __name__ == "__main__":
    KeypointNetGEKey(category='chair', graph='knn', knn=20, algorithm='peak', num_keys=10, visualization=True)
