import os
import json
import torch
import numpy as np
from sklearn.metrics import pairwise_distances_argmin

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .defaults import DefaultDataset
from .utils import naive_read_pcd
from .multiprocess import process_data_parallel

from .method.knngraph import KnnGraph
from .method.peakdetector import PeakDetector


@DATASETS.register_module()
class KeyPointNetDataset(DefaultDataset):
    def __init__(
            self,
            data_root="data/keypointnet",
            category="airplane",
            knn=20,
            pseudo=10,
            use_fps=True,
            n_farthest=10,
            save_record=True,
            **kwargs
    ):
        self.data_root = data_root
        self.category = category.lower()
        self.knn = knn
        self.pseudo = pseudo
        self.use_fps = use_fps
        self.n_farthest = n_farthest
        self.categories = []
        self.token2category = {}
        with open(os.path.join(self.data_root, "synsetoffset2category.txt"), "r") as f:
            for line in f:
                ls = line.strip().split()
                self.token2category[ls[1]] = len(self.categories)
                self.categories.append(ls[0].lower())
        self.category2token = {v: k for k, v in self.token2category.items()}

        super().__init__(data_root=self.data_root, **kwargs)
        self.keypoints = self.get_keypoints()
        self.rotation_groups = self.get_rotation_groups()
        self.max_keypoints = max(max(kp[1] for kp in keypoint) for keypoint in self.keypoints.values()) + 1

        record_name = f"keypointnet_{self.split}_{self.category}_knn{self.knn}_pseudo{self.pseudo}"
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
        if self.category == 'all':
            return data_list
        else:
            data_list = [entry for entry in data_list if entry.startswith(
                self.category2token[self.categories.index(self.category)]
            )]
            return data_list

    def get_keypoints(self):
        annots = json.load(open(os.path.join(self.data_root, "annotations", self.category + '.json')))
        keypoints = dict([(annot['model_id'], [(kp_info['pcd_info']['point_index'], kp_info['semantic_id'])
                                               for kp_info in annot['keypoints']]) for annot in annots])
        return keypoints

    def get_rotation_groups(self):
        annots = json.load(open(os.path.join(self.data_root, "annotations", self.category + '.json')))
        rotation_groups = dict([(annot['model_id'], annot['symmetries']['rotation']) for annot in annots])
        return rotation_groups

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]
        if data_name in self.data.keys():
            return self.data[data_name]
        else:
            class_id = data_name.split('-')[0]
            model_id = data_name.split('-')[-1].rstrip('\n')
            data_path = os.path.join(
                self.data_root, "pcds", class_id, model_id + ".pcd"
            )
            data = naive_read_pcd(data_path)
            coord, color = data[:, 0:3], data[:, 3:6]
            keypoints, semantic_id = map(list, zip(*self.keypoints[model_id]))

            local_maxima = None
            if self.split not in ("test", "all"):
                knn_graph = KnnGraph(coord, k=self.knn, use_fps=self.use_fps, n_farthest=self.n_farthest)
                G, geo_dists = knn_graph.get_graph_and_distances()
                detector = PeakDetector(G, geo_dists, coord, num_peaks=self.pseudo)
                local_maxima = detector.detect_peaks()

            label = -np.ones((self.max_keypoints,), dtype=np.int64)
            label[semantic_id] = keypoints

            labels = [label]
            idx2semid = {idx: semid for idx, semid in enumerate(semantic_id)}
            rotation_info = self.rotation_groups[model_id]
            keypoint_coords = np.array([coord[idx] for idx in label if idx >= 0])
            if len(rotation_info) > 0 and not self.category == 'chair':
                group_idxs = [info['kp_indexes'] for info in rotation_info]
                all_circle = not (False in [info['is_circle'] for info in rotation_info])

                group = group_idxs[0]
                group_points = np.array([coord[label[idx2semid[i]]] for i in group])
                rotation_center = np.mean(group_points, 0)
                rotation_center[1] = 0  # set y to 0
                rotation_multiplex = len(group)
                if all_circle:
                    rotation_multiplex = 36

                for i in range(1, rotation_multiplex):
                    angle = 2 * np.pi / rotation_multiplex * i
                    keypoints_rotated = (keypoint_coords - rotation_center) @ np.array(
                        [[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0],
                         [np.sin(angle), 0, np.cos(angle)]]).T + rotation_center
                    label_rotated = -np.ones_like(label, dtype=np.int64)
                    label_rotated[label >= 0] = pairwise_distances_argmin(keypoints_rotated, coord)

                    labels.append(label_rotated)

            cls_token = self.token2category[
                os.path.basename(os.path.dirname(data_path))
            ]

            kp_index = torch.zeros(coord.shape[0])
            labels = np.concatenate(labels)
            valid_labels = labels[labels != -1]
            kp_index[valid_labels] = 1
            kp_index = [kp_index]

            data_dict = dict(
                coord=coord,
                kp_index=kp_index,
                keypoints=keypoints,
                semantic_id=semantic_id,
                cls_token=cls_token,
                name=data_name.rstrip('\n'),
            )
            if self.split not in ("test", "all"):
                data_dict["pseudo_index"] = local_maxima
            return data_dict

    def prepare_test_data(self, idx):
        return self.prepare_train_data(idx)
