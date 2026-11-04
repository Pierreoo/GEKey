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
from method.visualizer.gt_visualizer import GTKeypointVisualizer
from method.metrics import coverage, inclusivity, mIoU


class KeypointNetGEKey:
    def __init__(
            self,
            split='all',
            data_root="Pointcept/data/keypointnet",
            root_cloud="Pointcept/data/keypointnet/pcds",
            root_annot="Pointcept/data/keypointnet/annotations",
            category="airplane",
            graph='knn',
            knn=20,
            algorithm='hmaxima',
            num_keys=10,
            visualization=False,
    ):
        self.split = split
        self.data_root = data_root
        self.root_cloud = root_cloud
        self.root_annot = root_annot
        self.category = category.lower()
        self.graph = graph
        self.algorithm = algorithm
        self.knn = knn
        self.num_keys = num_keys
        self.visualization = visualization
        self.categories = []
        self.token2category = {}

        with open(os.path.join(project_root, self.data_root, "synsetoffset2category.txt"), "r") as f:
            for line in f:
                ls = line.strip().split()
                self.token2category[ls[1]] = len(self.categories)
                self.categories.append(ls[0].lower())
        self.category2token = {v: k for k, v in self.token2category.items()}

        splits_path = os.path.join(self.data_root, 'splits')
        split_path = os.path.join(project_root, splits_path, "{}.txt".format(self.split))
        data_list = np.loadtxt(split_path, dtype="str")
        if len(data_list.shape) == 0:
            data_list = np.expand_dims(data_list, 0)
        if self.category == 'all':
            data_list = data_list
        else:
            data_list = [entry for entry in data_list if entry.startswith(
                self.category2token[self.categories.index(self.category)]
            )]

        gt_all = []
        pred_all = []
        coverage_all = []
        inclusivity_all = []
        for idx, data in enumerate(data_list):
            data_idx = idx % len(data_list)
            data_name = data_list[data_idx]
            category_number = data_name.split('-')[0]
            model_id = data_name.split('-')[-1].rstrip('\n')

            cloud_path = os.path.join(project_root, self.root_cloud, f"{category_number}")
            cloud_file = os.path.join(project_root, cloud_path, f"{model_id}.pcd")
            annot_file = os.path.join(project_root, self.root_annot, f"{category}.json")

            cloud = o3d.io.read_point_cloud(cloud_file)
            cloud.paint_uniform_color([0.5, 0.5, 0.5])
            coord = np.asarray(cloud.points)

            gt_visualizer = GTKeypointVisualizer(annot_file, model_id, cloud)
            gt_spheres = gt_visualizer.get_gt_spheres()
            gt_keypoints = np.array([kp.get_center() for kp in gt_spheres])
            gt_keypoints = torch.from_numpy(gt_keypoints).float()

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

            mIoU_score = mIoU(gt_keypoints.unsqueeze(0), pred_keypoints.unsqueeze(0), threshold=0.1)
            print('mIoU: {:.2f}'.format(mIoU_score))

            gt_all.append(gt_keypoints)
            pred_all.append(pred_keypoints)

        mIoU_score = mIoU(gt_all, pred_all, threshold=0.1)

        print('Avg. inclusivity: {:.2f}'.format(np.mean(inclusivity_all)))
        print('Avg. coverage: {:.2f}'.format(np.mean(coverage_all)))
        print('Avg. mIoU: {:.2f}'.format(mIoU_score))


if __name__ == "__main__":
    KeypointNetGEKey(category='airplane', graph='knn', knn=20, algorithm='peak', num_keys=10, visualization=True)