import os
import torch
import numpy as np
import pickle

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .defaults import DefaultDataset
from .utils import naive_read_smpl
from .multiprocess import process_data_parallel

from .method.knngraph import KnnGraph
from .method.peakdetector import PeakDetector


@DATASETS.register_module()
class SMPLDataset(DefaultDataset):
    def __init__(
            self,
            data_root="data/smpl",
            knn=20,
            pseudo=10,
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

        super().__init__(data_root=self.data_root, **kwargs)

        record_name = f"smpl_{self.split}_knn{self.knn}_pseudo{self.pseudo}"
        record_dir = os.path.join(self.data_root, "records")
        record_path = os.path.join(record_dir, f"{record_name}.pth")
        os.makedirs(record_dir, exist_ok=True)
        logger = get_root_logger()
        if os.path.isfile(record_path):
            logger.info(f"Loading record: {record_name} ...")
            self.data = torch.load(record_path, weights_only=False)
        else:
            logger.info(f"Preparing record: {record_name} ...")
            self.data = {}
            self.data = process_data_parallel(self, logger)
            if save_record:
                os.makedirs(os.path.dirname(record_path), exist_ok=True)
                torch.save(self.data, record_path)

    def get_data_list(self):
        splits_path = os.path.join(self.data_root, 'splits')
        split_path = os.path.join(splits_path, "{}.txt".format(self.split))
        data_list = np.loadtxt(split_path, dtype="str")
        if len(data_list.shape) == 0:
            data_list = np.expand_dims(data_list, 0)
        return data_list

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]

        # Return from in-memory cache if available
        if data_name in self.data:
            return self.data[data_name]

        model_id = data_name
        data_path = os.path.join(self.data_root, "pcds", model_id + ".pcd")

        coord = naive_read_smpl(data_path)

        graphs_dir = os.path.join(self.data_root, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        file_path = os.path.join(graphs_dir, f"{model_id}.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data_dict = pickle.load(f)
            self.data[data_name] = data_dict
            return data_dict

        local_maxima = None
        if self.split not in ("test", "all"):
            knn_graph = KnnGraph(coord, k=self.knn, use_fps=self.use_fps, n_farthest=self.n_farthest)
            G, geo_dists = knn_graph.get_graph_and_distances()

            detector = PeakDetector(G, geo_dists, coord, num_peaks=self.pseudo)
            local_maxima = detector.detect_peaks()

        data_dict = {
            "coord": coord,
            "name": data_name.rstrip('\n'),
        }

        if self.split not in ("test", "all"):
            data_dict["pseudo_index"] = local_maxima

        with open(file_path, "wb") as f:
            pickle.dump(data_dict, f)

        self.data[data_name] = data_dict

        return data_dict

    def prepare_test_data(self, idx):
        return self.prepare_train_data(idx)
