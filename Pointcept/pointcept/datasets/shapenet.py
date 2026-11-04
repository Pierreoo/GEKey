import os
import random
import h5py
import torch
import numpy as np
import open3d as o3d

from .builder import DATASETS
from .defaults import DefaultDataset
from .multiprocess import process_data_parallel
from pointcept.utils.logger import get_root_logger

from .method.knngraph import KnnGraph
from .method.peakdetector import PeakDetector


@DATASETS.register_module()
class ShapeNetDataset(DefaultDataset):
    def __init__(
            self,
            data_root="data/shapenet",
            split="train",
            save_record=False,
            normalize=False,
            include_label=True,
            subclasses=tuple(range(40)),
            sample=None,
            knn=20,
            pseudo=10,
            use_fps=True,
            n_farthest=100,
            noise=False,
            sigma=0.20,
            downsample=False,
            down_points=64,
            **kwargs
    ):
        self.data_root = data_root
        self.split = split
        self.normalize = normalize
        self.include_label = include_label
        self.subclasses = subclasses
        self.sample = sample
        self.knn = knn
        self.pseudo = pseudo
        self.use_fps = use_fps
        self.n_farthest = n_farthest
        self.noise = noise
        self.sigma = sigma
        self.downsample = downsample
        self.down_points = down_points
        self.data, self.labels = self.load_data()
        self.data_list = self.get_data_list()
        super().__init__(data_root=self.data_root, **kwargs)

        record_name = f"shapenet_{self.split}_{'_'.join(str(s) for s in self.subclasses)}_knn{self.knn}_pseudo{self.pseudo}"
        record_path = os.path.join(self.data_root, "records", f"{record_name}.pth")
        logger = get_root_logger()
        if os.path.isfile(record_path):
            logger.info(f"Loading record: {record_name} ...")
            self.data_dict = torch.load(record_path, weights_only=False)
        else:
            logger.info(f"Preparing record: {record_name} ...")
            self.data_dict = {}
            self.data_dict = process_data_parallel(self, logger)
            if save_record:
                os.makedirs(os.path.dirname(record_path), exist_ok=True)
                torch.save(self.data_dict, record_path)

    def load_h5(self, h5_filename, normalize=False, include_label=False):
        with h5py.File(h5_filename, 'r') as f:
            data = f['data'][:]
            if normalize:
                dmin = data.min(axis=1, keepdims=True).min(axis=-1, keepdims=True)
                dmax = data.max(axis=1, keepdims=True).max(axis=-1, keepdims=True)
                data = (data - dmin) / (dmax - dmin)
                data = 2.0 * (data - 0.5)
            if include_label:
                label = f['label'][:]
                return data, label
            else:
                return data, None

    def walk_files(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".h5"):
                    yield os.path.join(root, file)

    def load_data(self):
        parent = os.path.join(self.data_root, self.split)
        lazy = map(lambda x: self.load_h5(x, self.normalize, self.include_label), self.walk_files(parent))
        if self.include_label:
            xy = tuple(lazy)
            x = [x for x, y in xy]
            y = [y for x, y in xy]
            x = np.concatenate(x)
            y = np.concatenate(y)
            xf = []
            yf = []
            for xp, yp in zip(x, y):
                if yp[0] in self.subclasses:
                    if self.sample is None:
                        xf.append(xp)
                    else:
                        xf.append(random.choices(xp, k=self.sample))
                    yf.append(np.eye(len(self.subclasses))[self.subclasses.index(yp[0])])
            return np.array(xf), np.array(yf)
        return np.concatenate(tuple(lazy))

    def get_data_list(self):
        data_list = list(range(len(self.data)))
        return data_list

    def get_data(self, idx):
        data_name = self.data_list[idx]
        if data_name in self.data_dict.keys():
            return self.data_dict[data_name]
        else:
            coord = self.data[idx]
            label = self.labels[idx]

            if self.noise:
                noise = self.sigma * np.random.randn(*coord.shape)
                coord = coord + noise

            if self.downsample:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(coord)
                downpcd_farthest = pcd.farthest_point_down_sample(self.down_points)
                coord = np.asarray(downpcd_farthest.points)

            knn_graph = KnnGraph(coord, k=self.knn, use_fps=self.use_fps, n_farthest=self.n_farthest)
            G, geo_dists = knn_graph.get_graph_and_distances()
            detector = PeakDetector(G, geo_dists, coord, num_peaks=self.pseudo)
            local_maxima = detector.detect_peaks()

            data_dict = dict(
                coord=coord,
                pseudo_index=local_maxima,
                kp_index=label,
                cls_token=self.subclasses,
                name=f"{self.subclasses}-sample{idx}",
            )
            return data_dict

    def prepare_test_data(self, idx):
        return self.prepare_train_data(idx)