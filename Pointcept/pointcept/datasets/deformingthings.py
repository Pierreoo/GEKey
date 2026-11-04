import os
import torch
import numpy as np
import pickle
import open3d as o3d

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .defaults import DefaultDataset
from .multiprocess import process_frames_parallel

from .method.knngraph import KnnGraph
from .method.peakdetector import PeakDetector


@DATASETS.register_module()
class DeformingThingsDataset(DefaultDataset):
    def __init__(
            self,
            data_root="data/deformingthings",
            knn=20,
            pseudo=12,
            use_fps=True,
            n_farthest=100,
            save_record=True,
            **kwargs
    ):
        self.data_root = data_root
        self.knn = knn
        self.pseudo = pseudo
        self.use_fps = use_fps
        self.n_farthest = n_farthest
        self._cache_queue = []
        super().__init__(data_root=self.data_root, **kwargs)

        record_name = f"deformingthings_{self.split}_knn{self.knn}_pseudo{self.pseudo}_frame"
        record_dir = os.path.join(self.data_root, "records", record_name)
        self.record_dir = record_dir
        os.makedirs(record_dir, exist_ok=True)
        logger = get_root_logger()

        self.data_list = self.get_data_list()
        all_exist = all(
            os.path.isfile(self._get_frame_path(data_name))
            for data_name in self.data_list
        )
        if all_exist:
            logger.info(f"All {len(self.data_list)} frame files exist")
            logger.info(f"Frames will be loaded on-demand during training")
            self.data = {}
        elif not save_record:
            logger.info("save_record=False: records will not be built; "
                        "frames are computed on demand during training")
            self.data = {}
        else:
            logger.info(f"Preparing records in: {record_dir} ...")
            logger.info(f"Processing {len(self.data_list)} frames...")
            process_frames_parallel(self, logger)
            logger.info(f"Data will be loaded on-demand during training")
            self.data = {}

    def _graph_path(self, seq_name, frame_info):
        """Path for the cached peaks pickle of a single frame."""
        params = f"knn{self.knn}_pseudo{self.pseudo}_fps{int(self.use_fps)}_n{self.n_farthest}"
        return os.path.join(self.data_root, "graphs", params, seq_name, f"{frame_info}.pkl")

    def _get_frame_path(self, data_name):
        """Get the file path for a single frame's saved data."""
        # Replace '/' with '_' to create a valid filename
        safe_name = data_name.replace('/', '_')
        return os.path.join(self.record_dir, f"{safe_name}.pth")

    def get_data_list(self):
        """Load data list from split files."""
        splits_path = os.path.join(self.data_root, 'splits')
        split_path = os.path.join(splits_path, "{}.txt".format(self.split))

        if not os.path.exists(split_path):
            available = sorted(f[:-4] for f in os.listdir(splits_path)) if os.path.isdir(splits_path) else []
            raise FileNotFoundError(
                f"Split file not found: {split_path}\n"
                f"Available splits: {available if available else 'none'}"
            )

        data_list = np.loadtxt(split_path, dtype="str")
        if len(data_list.shape) == 0:
            data_list = np.expand_dims(data_list, 0)

        expanded_list = []
        for seq_name in data_list:
            pcd_dir = os.path.join(self.data_root, "pcds", seq_name)
            if os.path.exists(pcd_dir):
                pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
                for pcd_file in pcd_files:
                    # Format: {frame_idx}_{seq_name}.pcd
                    expanded_list.append(f"{seq_name}/{pcd_file[:-4]}")

        return np.array(expanded_list)

    def get_data_single(self, data_name):
        """
        Load a single data sample (used during initial processing).
        Separated from get_data to avoid cache lookups during processing.
        """
        # Parse data_name: "seq_name/frame_idx_seq_name"
        parts = data_name.split('/')
        seq_name = parts[0]
        frame_info = parts[1]  # e.g., "0_bear3EP_Agression"

        pcd_path = os.path.join(
            self.data_root, "pcds", seq_name, f"{frame_info}.pcd"
        )

        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"Point cloud not found: {pcd_path}")

        pcd_o3d = o3d.io.read_point_cloud(pcd_path)
        coord = np.asarray(pcd_o3d.points, dtype=np.float32)

        data_dict = dict(
            coord=coord,
            name=data_name.rstrip('\n'),
        )

        if self.split not in ("test", "all"):
            graph_path = self._graph_path(seq_name, frame_info)
            if os.path.exists(graph_path):
                with open(graph_path, "rb") as f:
                    graph_data = pickle.load(f)
                pseudo_index = graph_data["peaks"]
            else:
                knn_graph = KnnGraph(coord, k=self.knn, use_fps=self.use_fps, n_farthest=self.n_farthest)
                G, geo_dists = knn_graph.get_graph_and_distances()
                detector = PeakDetector(G, geo_dists, coord, num_peaks=self.pseudo)
                pseudo_index = detector.detect_peaks()

                os.makedirs(os.path.dirname(graph_path), exist_ok=True)

                with open(graph_path, "wb") as f:
                    graph_data = {
                        'peaks': pseudo_index,
                    }
                    pickle.dump(graph_data, f)

            data_dict["pseudo_index"] = pseudo_index

        return data_dict

    def _fix_data_types(self, frame_data):
        """Fix data types for a single frame."""
        if "pseudo_index" in frame_data and isinstance(frame_data["pseudo_index"], list):
            frame_data["pseudo_index"] = np.array(frame_data["pseudo_index"])

        return frame_data

    def get_data(self, idx):
        """Load a single data sample with LRU cache management."""
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]

        if data_name in self.data:
            if data_name in self._cache_queue:
                self._cache_queue.remove(data_name)
            self._cache_queue.append(data_name)
            return self.data[data_name]

        frame_path = self._get_frame_path(data_name)

        if os.path.exists(frame_path):
            frame_data = torch.load(frame_path, weights_only=False)
            frame_data = self._fix_data_types(frame_data)

            self.data[data_name] = frame_data
            self._cache_queue.append(data_name)

            return frame_data

        return self.get_data_single(data_name)

    def prepare_test_data(self, idx):
        """Prepare test data (same as train for this dataset)."""
        return self.prepare_train_data(idx)