import os
import sys
import torch
import numpy as np
import open3d as o3d

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from method.graphs.knngraph import KnnGraph
from method.detectors.hmaximadetector import HMaximaDetector
from method.detectors.peakdetector import PeakDetector
from method.detectors.fpsdetector import FPSDetector
from method.detectors.issdetector import ISSDetector
from method.visualizer.gt_visualizer import GTKeypointVisualizer
from method.metrics import coverage, mIoU


class KeypointNetGEKey:
    def __init__(
            self,
            split='all',
            data_root="Pointcept/data/keypointnet",
            root_cloud="Pointcept/data/keypointnet/pcds",
            root_annot="Pointcept/data/keypointnet/annotations",
            category="airplane",
            knn=20,
            algorithm='hmaxima',
            num_keys=10,
            miou_threshold=0.1,
            visualization=False,
    ):
        self.split = split
        self.data_root = data_root
        self.root_cloud = root_cloud
        self.root_annot = root_annot
        self.category = category.lower()
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
        data_list = [entry for entry in data_list if entry.startswith(
            self.category2token[self.categories.index(self.category)]
        )]

        gt_all = []
        pred_all = []
        coverage_all = []

        for idx, data in enumerate(data_list):
            data_idx = idx % len(data_list)
            data_name = data_list[data_idx]
            category_number = data_name.split('-')[0]
            model_id = data_name.split('-')[-1].rstrip('\n')

            cloud_path = os.path.join(project_root, self.root_cloud, f"{category_number}")
            cloud_file = os.path.join(project_root, cloud_path, f"{model_id}.pcd")
            annot_file = os.path.join(project_root, self.root_annot, f"{self.category}.json")

            cloud = o3d.io.read_point_cloud(cloud_file)
            coord = np.asarray(cloud.points)

            gt_visualizer = GTKeypointVisualizer(annot_file, model_id, cloud)
            gt_spheres = gt_visualizer.get_gt_spheres()
            gt_keypoints = np.array([kp.get_center() for kp in gt_spheres])
            gt_keypoints = torch.from_numpy(gt_keypoints).float()

            knn_graph = KnnGraph(coord, k=self.knn, use_fps=True, n_farthest=100)
            G, geo_dists = knn_graph.get_graph_and_distances()

            if self.algorithm == 'peak':
                detector = PeakDetector(G, geo_dists, coord, num_peaks=self.num_keys)
                maxima = detector.get_peak_spheres()
            elif self.algorithm == 'hmaxima':
                detector = HMaximaDetector(G, geo_dists, coord, target_maxima=self.num_keys)
                maxima = detector.get_hmaxima_spheres()
            elif self.algorithm == 'fps':
                detector = FPSDetector(coord, num_samples=self.num_keys)
                maxima = detector.get_fps_spheres()
            elif self.algorithm == 'iss':
                detector = ISSDetector(cloud, coord, num_keys=self.num_keys)
                maxima = detector.get_iss_spheres()
            else:
                print(f"Unsupported algorithm: '{self.algorithm}', use 'hmaxima', 'peak', 'fps' or 'iss'")

            centers = np.array([s.get_center() for s in maxima])
            pred_keypoints = torch.from_numpy(centers).float()
            coord = torch.from_numpy(coord).float()
            coverage_score = coverage(pred_keypoints.unsqueeze(0), coord.unsqueeze(0))
            coverage_all.append(coverage_score)

            gt_all.append(gt_keypoints)
            pred_all.append(pred_keypoints)

        mIoU_score = mIoU(gt_all, pred_all, threshold=miou_threshold)

        self.coverage_all = coverage_all
        self.mIoU_score = mIoU_score

        print(f"Avg. coverage: {np.mean(coverage_all):.2f}")
        print(f"Avg. mIoU: {mIoU_score:.2f}")


if __name__ == "__main__":
    knn, algorithm, miou_threshold = 20, 'peak', 0.1

    categories = [
        "airplane", "bed", "bottle", "cap",
        "car", "chair", "guitar", "helmet",
        "knife", "motorcycle", "mug", "table",
        "vessel"
    ]

    all_coverage = []
    all_das = []
    all_miou = []

    results_folder = os.path.join(project_root, "method/results")
    if algorithm == 'peak' and miou_threshold == 0.1:
        output_path = os.path.join(results_folder, "k-parameter", f"results_all_k{knn}.txt")
    else:
        name = 'gekey_pl' if algorithm == 'peak' else algorithm
        output_path = os.path.join(results_folder, "coverage", f"results_{miou_threshold:.2f}_{name}.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("Category-wise Evaluation Results\n")
        f.write("=" * 40 + "\n")

        for category in categories:
            print(f"\n=== Evaluating Category: {category} ===")
            evaluator = KeypointNetGEKey(
                split='all',
                category=category,
                knn=knn,
                algorithm=algorithm,
                num_keys=10,
                miou_threshold=miou_threshold,
                visualization=False
            )

            avg_coverage = np.mean(evaluator.coverage_all)
            avg_miou = evaluator.mIoU_score

            all_coverage.append(avg_coverage)
            all_miou.append(avg_miou)

            print(f"Avg. coverage: {avg_coverage:.2f}")
            print(f"Avg. mIoU: {avg_miou:.2f}")

            f.write(f"{category:<12} | Coverage: {avg_coverage:.2f} | mIoU: {avg_miou:.2f}\n")

        f.write("=" * 40 + "\n")
        f.write(f"Overall Avg   | Coverage: {np.mean(all_coverage):.2f} | mIoU: {np.mean(all_miou):.2f}\n")
