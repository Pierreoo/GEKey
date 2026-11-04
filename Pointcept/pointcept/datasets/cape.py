import os
import torch
import numpy as np
import pickle

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .defaults import DefaultDataset
from .multiprocess import process_data_parallel

from .method.knngraph import KnnGraph
from .method.peakdetector import PeakDetector


@DATASETS.register_module()
class CAPEDataset(DefaultDataset):
    def __init__(
        self,
        data_root="data/cape",
        knn=20,
        pseudo=12,
        use_fps=True,
        n_farthest=100,
        normalize=False,
        save_record=True,
        **kwargs,
    ):
        self.data_root = data_root
        self.knn = knn
        self.pseudo = pseudo
        self.use_fps = use_fps
        self.n_farthest = n_farthest
        self.normalize = normalize

        super().__init__(data_root=self.data_root, **kwargs)

        record_name = f"cape_{self.split}_knn{self.knn}_pseudo{self.pseudo}"
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
                torch.save(self.data, record_path)

    @staticmethod
    def parse_model_id(model_id: str) -> tuple[str, str, str]:
        subject, rest = model_id.split("_", 1)
        sequence, frame = rest.rsplit(".", 1)
        return subject, sequence, frame

    def _pcd_path(self, model_id: str) -> str:
        subject, sequence, frame = self.parse_model_id(model_id)
        filename = f"{subject}_{sequence}_{frame}.npy"
        return os.path.join(self.data_root, "pcds", filename)

    def _graph_path(self, model_id: str) -> str:
        safe = model_id.replace("/", "_")
        subject, sequence, _ = self.parse_model_id(model_id)
        params = f"knn{self.knn}_pseudo{self.pseudo}_fps{int(self.use_fps)}_n{self.n_farthest}"
        return os.path.join(self.data_root, "graphs", params, subject, sequence, f"{safe}.pkl")

    def get_data_list(self):
        split_path = os.path.join(self.data_root, "splits", f"{self.split}.txt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Split file not found: {split_path}\n"
                f"Run splits.py first to generate train/val/test.txt."
            )

        data_list = np.loadtxt(split_path, dtype="str")
        if data_list.ndim == 0:
            data_list = np.expand_dims(data_list, 0)

        logger = get_root_logger()
        valid, missing = [], []
        for model_id in data_list:
            if os.path.isfile(self._pcd_path(model_id)):
                valid.append(model_id)
            else:
                missing.append(model_id)

        if missing:
            logger.warning(
                f"{len(missing)} entries in {self.split}.txt have no matching "
                f".npy file and will be skipped. First missing: {missing[0]}"
            )

        return np.array(valid)

    @staticmethod
    def _normalize_pc(pc: np.ndarray) -> np.ndarray:
        pc = pc - pc.mean(axis=0)
        pc /= np.max(np.linalg.norm(pc, axis=-1))
        return pc / 2.0

    def get_data(self, idx: int) -> dict:
        data_name = self.data_list[idx % len(self.data_list)]

        if data_name in self.data:
            return self.data[data_name]

        pcd_path = self._pcd_path(data_name)
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"Point cloud not found: {pcd_path}")

        coord = np.load(pcd_path).astype(np.float32)
        if self.normalize:
            coord = self._normalize_pc(coord)

        data_dict = dict(coord=coord, name=data_name)

        if self.split not in ("test", "all"):
            graph_path = self._graph_path(data_name)
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
                    pickle.dump({"peaks": pseudo_index}, f)

            data_dict["pseudo_index"] = pseudo_index

        return data_dict

    def prepare_test_data(self, idx: int) -> dict:
        return self.prepare_train_data(idx)
