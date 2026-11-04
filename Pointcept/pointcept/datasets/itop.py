import os
import pickle

import numpy as np
import torch

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .defaults import DefaultDataset
from .multiprocess import process_data_parallel
from .method.knngraph import KnnGraph
from .method.peakdetector import PeakDetector


@DATASETS.register_module()
class ITOPDataset(DefaultDataset):
    """
    Dataloader for the ITOP dataset (side or top view).

    Expected on-disk layout (produced by the preprocessing script):
        <data_root>/
            pcds/
                <frame_id>.npy      # (2048, 3) float32 point cloud
            splits/
                train.txt
                val.txt
                test.txt
            graphs/                 # auto-created on first run
                <frame_id>.pkl
            records/                # auto-created on first run
                itop_<split>_knn<k>_pseudo<p>.pth

    Each line of a split .txt file is one frame_id, e.g. ``0001_00001``.

    The frame_id format mirrors the ITOP HDF5 ``id`` field:
        ``<sequence>_<frame>``  (both zero-padded strings)
    """

    def __init__(
        self,
        data_root="data/itop",
        knn: int = 20,
        pseudo: int = 12,
        use_fps: bool = True,
        n_farthest: int = 100,
        normalize: bool = False,
        save_record: bool = True,
        **kwargs,
    ):
        self.data_root = data_root
        self.knn = knn
        self.pseudo = pseudo
        self.use_fps = use_fps
        self.n_farthest = n_farthest
        self.normalize = normalize

        # DefaultDataset.__init__ sets self.split and calls get_data_list()
        super().__init__(data_root=self.data_root, **kwargs)

        # Record cache
        record_name = f"itop_{self.split}_knn{self.knn}_pseudo{self.pseudo}"
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

    # ID helpers

    @staticmethod
    def parse_frame_id(frame_id: str) -> tuple[str, str]:
        """
        Split ``'<sequence>_<frame>'`` into ``(sequence, frame)``.

        Example: ``'0001_00042'`` -> ``('0001', '00042')``
        """
        sequence, frame = frame_id.split("_", 1)
        return sequence, frame

    def _pcd_path(self, frame_id: str) -> str:
        """Absolute path to the pre-processed .npy point cloud."""
        return os.path.join(self.data_root, "pcds", f"{frame_id}.npy")

    def _graph_path(self, frame_id: str) -> str:
        """Absolute path for the cached KNN-graph / peaks pickle."""
        params = f"knn{self.knn}_pseudo{self.pseudo}_fps{int(self.use_fps)}_n{self.n_farthest}"
        return os.path.join(self.data_root, "graphs", params, f"{frame_id}.pkl")

    # Split loading

    def get_data_list(self) -> np.ndarray:
        splits_path = os.path.join(self.data_root, "splits")
        split_path = os.path.join(splits_path, f"{self.split}.txt")
        if not os.path.exists(split_path):
            available = sorted(f[:-4] for f in os.listdir(splits_path)) if os.path.isdir(splits_path) else []
            raise FileNotFoundError(
                f"Split file not found: {split_path}\n"
                f"Available splits: {available if available else 'none'}"
            )

        data_list = np.loadtxt(split_path, dtype="str")
        if data_list.ndim == 0:
            data_list = np.expand_dims(data_list, 0)

        logger = get_root_logger()
        valid, missing = [], []
        for frame_id in data_list:
            if os.path.isfile(self._pcd_path(frame_id)):
                valid.append(frame_id)
            else:
                missing.append(frame_id)

        if missing:
            logger.warning(
                f"{len(missing)} entries in {self.split}.txt have no matching "
                f".npy file and will be skipped. First missing: {missing[0]}"
            )

        return np.array(valid)

    # Optional unit-sphere normalization

    @staticmethod
    def _normalize_pc(pc: np.ndarray) -> np.ndarray:
        pc = pc - pc.mean(axis=0)
        pc /= np.max(np.linalg.norm(pc, axis=-1))
        return pc / 2.0

    # Core data-loading logic

    def get_data(self, idx: int) -> dict:
        frame_id = self.data_list[idx % len(self.data_list)]

        # Return from in-memory cache if already processed
        if frame_id in self.data:
            return self.data[frame_id]

        pcd_path = self._pcd_path(frame_id)
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"Point cloud not found: {pcd_path}")

        coord = np.load(pcd_path).astype(np.float32)  # (2048, 3)
        if self.normalize:
            coord = self._normalize_pc(coord)

        # Store the sequence/frame breakdown for downstream use
        sequence, frame = self.parse_frame_id(frame_id)
        data_dict = dict(
            coord=coord,
            name=frame_id,
            sequence=sequence,
            frame=frame,
        )

        # Pseudo-label peaks are only needed during training / validation
        if self.split not in ("test", "all"):
            graph_path = self._graph_path(frame_id)

            if os.path.exists(graph_path):
                with open(graph_path, "rb") as f:
                    graph_data = pickle.load(f)
                pseudo_index = graph_data["peaks"]
            else:
                knn_graph = KnnGraph(coord, k=self.knn, use_fps=self.use_fps, n_farthest=self.n_farthest)
                G, geo_dists = knn_graph.get_graph_and_distances()
                detector = PeakDetector(
                    G, geo_dists, coord, num_peaks=self.pseudo
                )
                pseudo_index = detector.detect_peaks()

                os.makedirs(os.path.dirname(graph_path), exist_ok=True)
                with open(graph_path, "wb") as f:
                    pickle.dump({"peaks": pseudo_index}, f)

            data_dict["pseudo_index"] = pseudo_index

        return data_dict

    def prepare_test_data(self, idx: int) -> dict:
        return self.prepare_train_data(idx)