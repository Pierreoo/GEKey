import os
import sys
import time
import torch
import numpy as np
import open3d as o3d
from multiprocessing import Pool

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from method.graphs.knngraph import KnnGraph
from method.detectors.peakdetector import PeakDetector
from method.visualizer.gt_visualizer import GTKeypointVisualizer
from method.metrics import coverage, inclusivity, mIoU


def process_single_sample(args):
    data_name, category, root_cloud, root_annot, knn, algorithm, num_keys, use_fps, n_farthest = args

    category_number = data_name.split('-')[0]
    model_id = data_name.split('-')[-1].rstrip('\n')

    cloud_path = os.path.join(project_root, root_cloud, f"{category_number}")
    cloud_file = os.path.join(project_root, cloud_path, f"{model_id}.pcd")
    annot_file = os.path.join(project_root, root_annot, f"{category}.json")

    cloud = o3d.io.read_point_cloud(cloud_file)
    cloud.paint_uniform_color([0.5, 0.5, 0.5])
    coord = np.asarray(cloud.points)

    gt_visualizer = GTKeypointVisualizer(annot_file, model_id, cloud)
    gt_spheres = gt_visualizer.get_gt_spheres()
    gt_keypoints = np.array([kp.get_center() for kp in gt_spheres])
    gt_keypoints = torch.from_numpy(gt_keypoints).float()

    time_graph = 0.0
    time_eccentricity = 0.0
    time_maxima = 0.0

    knn_graph = KnnGraph(coord, k=knn, use_fps=use_fps, n_farthest=n_farthest)
    time_graph += knn_graph.t_graph          # KNN build + component repair + CSR graph view
    time_eccentricity += knn_graph.t_ecc     # eccentricity geodesic pass (measured inside __init__)

    G, geo_dists = knn_graph.get_graph_and_distances()

    start = time.time()
    detector = PeakDetector(G, geo_dists, coord, num_peaks=num_keys)
    maxima = detector.get_peak_spheres(points=True, verbose=False)
    time_maxima += time.time() - start

    pred_keypoints = torch.from_numpy(maxima).float()
    coord = torch.from_numpy(coord).float()

    inclusivity_score = inclusivity(pred_keypoints, coord, threshold=0.1)
    coverage_score = coverage(pred_keypoints.unsqueeze(0), coord.unsqueeze(0))
    mIoU_score = mIoU(gt_keypoints.unsqueeze(0), pred_keypoints.unsqueeze(0), threshold=0.1)

    return (
        gt_keypoints,
        pred_keypoints,
        coverage_score,
        torch.mean(inclusivity_score),
        mIoU_score,
        time_graph,
        time_eccentricity,
        time_maxima
    )


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
        use_fps=False,
        n_farthest=100,
        visualization=False,
        use_parallel=True
    ):
        self.split = split
        self.data_root = data_root
        self.root_cloud = root_cloud
        self.root_annot = root_annot
        self.category = category.lower()
        self.algorithm = algorithm
        self.knn = knn
        self.num_keys = num_keys
        self.use_fps = use_fps
        self.n_farthest = n_farthest
        self.visualization = visualization
        self.use_parallel = use_parallel

        self.categories = []
        self.token2category = {}
        with open(os.path.join(project_root, self.data_root, "synsetoffset2category.txt"), "r") as f:
            for line in f:
                ls = line.strip().split()
                self.token2category[ls[1]] = len(self.categories)
                self.categories.append(ls[0].lower())
        self.category2token = {v: k for k, v in self.token2category.items()}

        splits_path = os.path.join(self.data_root, "splits")
        split_path = os.path.join(project_root, splits_path, f"{self.split}.txt")
        data_list = np.loadtxt(split_path, dtype="str")
        if len(data_list.shape) == 0:
            data_list = np.expand_dims(data_list, 0)

        if self.category != "all":
            data_list = [
                entry for entry in data_list
                if entry.startswith(self.category2token[self.categories.index(self.category)])
            ]

        args_list = [
            (data, self.category, self.root_cloud, self.root_annot,
             self.knn, self.algorithm, self.num_keys, self.use_fps, self.n_farthest)
            for data in data_list
        ]

        if self.use_parallel:
            print(f"Processing {len(args_list)} samples in parallel...")
            with Pool(processes=os.cpu_count()) as pool:
                results = pool.map(process_single_sample, args_list)
        else:
            print(f"Processing {len(args_list)} samples sequentially...")
            results = [process_single_sample(args) for args in args_list]

        (
            gt_all,
            pred_all,
            coverage_all,
            inclusivity_all,
            mIoU_all,
            graph_times,
            eccentricity_times,
            maxima_times
        ) = zip(*results)

        self.coverage_all = np.array([float(c) for c in coverage_all])
        self.inclusivity_all = np.array([float(i) for i in inclusivity_all])
        self.mIoU_all = np.array([float(m) for m in mIoU_all])

        self.total_graph = sum(graph_times)
        self.total_eccentricity = sum(eccentricity_times)
        self.total_maxima = sum(maxima_times)

        self.avg_graph = np.mean(graph_times)
        self.avg_eccentricity = np.mean(eccentricity_times)
        self.avg_maxima = np.mean(maxima_times)


if __name__ == "__main__":
    n_farthest = 10  # 10 or 100

    categories = [
        "airplane", "bed", "bottle", "cap",
        "car", "chair", "guitar", "helmet",
        "knife", "motorcycle", "mug", "table",
        "vessel"
    ]

    results_folder = os.path.join(project_root, "method/results/runtime")
    os.makedirs(results_folder, exist_ok=True)
    output_path = os.path.join(results_folder, f"results_runtime_{n_farthest}.txt")

    with open(output_path, "w") as f:
        f.write("Category-wise Evaluation Results\n")
        f.write("=" * 70 + "\n\n")

        all_coverage = []
        all_miou = []
        all_total_graph = []
        all_total_eccentricity = []
        all_total_maxima = []
        all_avg_graph = []
        all_avg_eccentricity = []
        all_avg_maxima = []

        for category in categories:
            print(f"\n=== Evaluating Category: {category} ===")

            evaluator = KeypointNetGEKey(
                split='all',
                category=category,
                knn=20,
                algorithm='peak',
                num_keys=10,
                use_fps=True,
                n_farthest=n_farthest,
                visualization=False,
                use_parallel=False
            )

            avg_coverage = np.mean(evaluator.coverage_all)
            avg_mIoU = np.mean(evaluator.mIoU_all)
            all_coverage.append(avg_coverage)
            all_miou.append(avg_mIoU)
            all_total_graph.append(evaluator.total_graph)
            all_total_eccentricity.append(evaluator.total_eccentricity)
            all_total_maxima.append(evaluator.total_maxima)
            all_avg_graph.append(evaluator.avg_graph)
            all_avg_eccentricity.append(evaluator.avg_eccentricity)
            all_avg_maxima.append(evaluator.avg_maxima)

            f.write(f"{category.upper()}\n")
            f.write(f"Coverage: {avg_coverage:.4f}\n")
            f.write(f"mIoU:     {avg_mIoU:.4f}\n")
            f.write(f"Graph Time:      total={evaluator.total_graph:.4f}s   avg={evaluator.avg_graph:.4f}s\n")
            f.write(f"Eccentricity Time:    total={evaluator.total_eccentricity:.4f}s avg={evaluator.avg_eccentricity:.4f}s\n")
            f.write(f"Maxima Time: total={evaluator.total_maxima:.4f}s avg={evaluator.avg_maxima:.4f}s\n")
            f.write("-" * 70 + "\n")

        f.write("\nOVERALL SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Mean Coverage: {np.mean(all_coverage):.4f}\n")
        f.write(f"Mean mIoU:     {np.mean(all_miou):.4f}\n")
        f.write(f"Mean Total Graph Time:      {np.mean(all_total_graph):.4f}s\n")
        f.write(f"Mean Total Eccentricity Time:    {np.mean(all_total_eccentricity):.4f}s\n")
        f.write(f"Mean Total Maxima Time: {np.mean(all_total_maxima):.4f}s\n")
        f.write(f"Mean Avg Graph Time:        {np.mean(all_avg_graph):.4f}s\n")
        f.write(f"Mean Avg Eccentricity Time:      {np.mean(all_avg_eccentricity):.4f}s\n")
        f.write(f"Mean Avg Maxima Time:   {np.mean(all_avg_maxima):.4f}s\n")

    print("\nAll results written to:", output_path)
