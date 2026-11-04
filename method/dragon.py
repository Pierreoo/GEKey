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


class DragonGEKey:
    def __init__(
            self,
            split='all',
            data_root="Pointcept/data/",
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

        dragon = os.path.join(project_root, self.data_root, "dragon.obj")
        mesh = o3d.io.read_triangle_mesh(dragon)
        coord = np.asarray(mesh.vertices)

        pcmax = coord.max()
        pcmin = coord.min()
        coord = (coord - pcmin) / (pcmax - pcmin)
        coord = 2.0 * (coord - 0.5)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord)
        pcd_down = pcd.farthest_point_down_sample(10000)
        coord = np.asarray(pcd_down.points)

        knn_graph = KnnGraph(coord, k=self.knn, use_fps=True, n_farthest=100)
        G, geo_dists = knn_graph.get_graph_and_distances()

        if self.visualization:
            knn_graph.visualize()

        if algorithm == 'peak':
            detector = PeakDetector(G, geo_dists, coord, num_peaks=self.num_keys)
            maxima = detector.get_peak_spheres(radius=0.025)
            if visualization:
                heights = detector.compute_heights()
                detector.visualize_heights(heights)
        elif algorithm == 'hmaxima':
            detector = HMaximaDetector(G, geo_dists, coord, target_maxima=self.num_keys)
            maxima = detector.get_hmaxima_spheres()
        elif algorithm == 'fps':
            detector = FPSDetector(coord, num_samples=self.num_keys)
            maxima = detector.get_fps_spheres()
        else:
            print(f"Unsupported algorithm: '{algorithm}', use 'hmaxima', 'peak' or 'fps'")

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


if __name__ == "__main__":
    DragonGEKey(knn=20, algorithm='peak', num_keys=43, visualization=True)
