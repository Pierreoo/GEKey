import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.utils.data
from scipy.spatial import cKDTree
import open3d as o3d

from .defaults import create_ddp_model
from pointcept.engines.metrics.keypoint import KeypointMetrics
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    CategoryAverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
    fp_fn_geodist,
    iou_per_category
)
from collections import defaultdict


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class KeypointTesterMatch(TesterBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.metrics = KeypointMetrics()
        self.additional_metrics = True

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        save_path_pcd = os.path.join(self.cfg.save_path, "pcds")
        make_dirs(save_path)
        make_dirs(save_path_pcd)

        comm.synchronize()
        all_names = []
        all_pred_peaks_list = []
        all_gt_peaks_list = []
        all_coords = []
        all_coords_original = []
        all_gt_keypoints = []
        all_sematics = []

        for idx, input_dict in enumerate(self.test_loader):
            end = time.time()
            input_dict = input_dict[0]
            name = input_dict["name"]
            all_names.append(name)
            all_coords.append(input_dict["coord"].cpu().numpy())
            all_coords_original.append(input_dict["coord_original"].cpu().numpy())
            all_gt_keypoints.append(input_dict["keypoints"])
            all_sematics.append(input_dict["semantic_id"])

            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.model(input_dict)

            all_pred_peaks = output_dict["all_pred_peaks"]
            all_pred_peaks = torch.stack(all_pred_peaks, dim=0)

            label = input_dict["kp_index"][0].squeeze()
            gt_indices = torch.where(label == 1)[0]
            all_gt_peaks = input_dict["coord_original"][gt_indices].unsqueeze(0)

            all_pred_peaks_list.append(all_pred_peaks)
            all_gt_peaks_list.append(all_gt_peaks)

            pred_peaks_tensor = all_pred_peaks[0].cpu().numpy()
            coord = input_dict["coord"]
            tree = cKDTree(coord.cpu())
            dists, indices = tree.query(pred_peaks_tensor, k=1)
            pred_peaks_surface = coord[indices]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord.cpu().numpy())

            o3d.io.write_point_cloud(os.path.join(save_path_pcd, f"{name}.pcd"), pcd)
            np.save(os.path.join(save_path, f"{name}.npy"), pred_peaks_surface.cpu().numpy())

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                    name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()

        if comm.is_main_process():
            for i in range(11):
                dist_thresh = 0.01 * i

                all_coverage = []
                all_inclusivity = []
                all_pred_peaks_surface = []
                all_gt_peaks_np = []
                combined_preds_forward = []
                combined_preds_backward = defaultdict(list)

                for sample_idx, (pred_peaks, gt_peaks, coords, coords_original, gt_keypoints, semantics) in enumerate(
                        zip(all_pred_peaks_list, all_gt_peaks_list, all_coords, all_coords_original, all_gt_keypoints, all_sematics)):
                    pred_peaks_np = np.stack([p.cpu().numpy() for p in pred_peaks])
                    gt_peaks_np = np.stack([g.cpu().numpy() for g in gt_peaks])
                    keypoints_int = [k.item() if isinstance(k, torch.Tensor) else int(k) for k in gt_keypoints]
                    semantics_int = [s.item() if isinstance(s, torch.Tensor) else int(s) for s in semantics]

                    tree = cKDTree(coords)
                    dists, indices = tree.query(pred_peaks_np, k=1)
                    pred_peaks_surface = coords_original[indices]

                    all_pred_peaks_surface.append(pred_peaks_surface.squeeze(0))
                    all_gt_peaks_np.append(gt_peaks_np.squeeze(0))

                    if self.additional_metrics:
                        pred_peaks_tensor = torch.from_numpy(pred_peaks_np).float()
                        coords_tensor = torch.from_numpy(coords).float().unsqueeze(0)
                        coverage_score = self.metrics.coverage(pred_peaks_tensor, coords_tensor)
                        inclusivity_score = self.metrics.inclusivity(pred_peaks_tensor.squeeze(0), coords_tensor.squeeze(0), threshold=0.10)
                        all_coverage.append(coverage_score)
                        all_inclusivity.append(inclusivity_score)

                        pred_peaks_np = pred_peaks_np.squeeze(0)
                        gt_peaks_normalized = coords[keypoints_int]

                        tree_gt = cKDTree(gt_peaks_normalized)
                        dists_forward, indices_forward = tree_gt.query(pred_peaks_np, k=1)
                        pred_sem_ids = [semantics_int[idx] for idx in indices_forward]
                        combined_preds_forward.append(pred_sem_ids)

                        tree_pred = cKDTree(pred_peaks_np)
                        dists_backward, indices_backward = tree_pred.query(gt_peaks_normalized, k=1)
                        preds_backward = defaultdict(list)
                        for j in range(len(gt_peaks_normalized)):
                            sem_id = semantics_int[j]
                            preds_backward[sem_id].append(indices_backward[j])
                        for sem_id, indices in preds_backward.items():
                            combined_preds_backward[sem_id].extend(indices)

                miou_test = self.metrics.compute_iou(all_pred_peaks_surface, all_gt_peaks_np, dist_thresh)
                logger.info(f"Test Result @ dist {dist_thresh:.2f}: mIoU = {miou_test:.4f}")

            if self.additional_metrics:
                combined_preds_forward = np.array(combined_preds_forward)
                combined_preds_backward = dict(sorted(combined_preds_backward.items()))
                das_forward = self.metrics.fwd_alignment_scores(combined_preds_forward)
                das_backward = self.metrics.bwd_alignment_scores(combined_preds_backward)
                das_sm = ((das_backward + das_forward) / 2) * 100

            all_pred_peaks_tensor = torch.cat(all_pred_peaks_list, dim=0)
            all_pred_peaks_np_list = [p.cpu().numpy() for p in all_pred_peaks_tensor]
            das_sc3k = self.metrics.DAS_unsupervised(all_pred_peaks_np_list)

            if self.additional_metrics:
                logger.info("Test Result: inclusivity = {:.2f}".format(np.mean(all_inclusivity)))
                logger.info("Test Result: coverage = {:.2f}".format(np.mean(all_coverage)))
                logger.info("Test Result: DAS_SM = {:.2f}".format(das_sm))

            logger.info(f"Test Result: DAS = {das_sc3k:.2f}")

            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class KeypointTesterSMPL(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        save_path_pcd = os.path.join(self.cfg.save_path, "pcds")
        make_dirs(save_path)
        make_dirs(save_path_pcd)

        comm.synchronize()
        all_names = []
        all_pred_peaks_list = []
        all_coords = []
        all_coords_original = []

        for idx, input_dict in enumerate(self.test_loader):
            end = time.time()
            input_dict = input_dict[0]
            name = input_dict["name"]
            all_names.append(name)
            all_coords.append(input_dict["coord"].cpu().numpy())
            all_coords_original.append(input_dict["coord_original"].cpu().numpy())

            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.model(input_dict)

            all_pred_peaks = output_dict["all_pred_peaks"]
            all_pred_peaks = torch.stack(all_pred_peaks, dim=0)

            all_pred_peaks_list.append(all_pred_peaks)

            pred_peaks_tensor = all_pred_peaks[0].cpu().numpy()
            coord = input_dict["coord"]
            tree = cKDTree(coord.cpu())
            dists, indices = tree.query(pred_peaks_tensor, k=1)
            pred_peaks_surface = coord[indices]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord.cpu().numpy())

            o3d.io.write_point_cloud(os.path.join(save_path_pcd, f"{name}.pcd"), pcd)
            np.save(os.path.join(save_path, f"{name}.npy"), pred_peaks_surface.cpu().numpy())

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                    name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                )
            )

        logger.info("Syncing ...")
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class KeypointTesterDeformingThings(TesterBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.metrics = KeypointMetrics()

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        save_path_pcd = os.path.join(self.cfg.save_path, "pcds")
        os.makedirs(save_path_pcd, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        comm.synchronize()
        all_names = []
        all_pred_peaks_list = []
        all_coords = []

        for idx, input_dict in enumerate(self.test_loader):
            end = time.time()
            input_dict = input_dict[0]
            name = input_dict["name"]
            all_names.append(name)
            all_coords.append(input_dict["coord"].cpu().numpy())

            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.model(input_dict)

            all_pred_peaks = output_dict["all_pred_peaks"]
            all_pred_peaks = torch.stack(all_pred_peaks, dim=0)

            all_pred_peaks_list.append(all_pred_peaks)

            pred_peaks_tensor = all_pred_peaks[0].cpu().numpy()
            coord = input_dict["coord"]
            tree = cKDTree(coord.cpu())
            dists, indices = tree.query(pred_peaks_tensor, k=1)
            pred_peaks_surface = coord[indices]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord.cpu().numpy())

            parts = name.split("/")
            sequence_name = parts[0]
            frame_num = parts[1].split("_")[0]
            safe_name = f"{sequence_name}_frame{frame_num}"
            pcd_path = os.path.join(save_path_pcd, f"{safe_name}.pcd")
            npy_path = os.path.join(save_path, f"{safe_name}.npy")

            o3d.io.write_point_cloud(pcd_path, pcd)
            np.save(npy_path, pred_peaks_surface.cpu().numpy())

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                    name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                )
            )

        if comm.is_main_process():
            all_coverage = []
            all_inclusivity = []
            all_consistency = []

            for sample_idx, (pred_peaks, coords) in enumerate(
                    zip(all_pred_peaks_list, all_coords)):
                pred_peaks_np = np.stack([p.cpu().numpy() for p in pred_peaks])

                pred_peaks_tensor = torch.from_numpy(pred_peaks_np).float()
                coords_tensor = torch.from_numpy(coords).float().unsqueeze(0)
                coverage_score = self.metrics.coverage(pred_peaks_tensor, coords_tensor)
                inclusivity_score = self.metrics.inclusivity(pred_peaks_tensor.squeeze(0), coords_tensor.squeeze(0), threshold=0.05)
                all_coverage.append(coverage_score)
                all_inclusivity.append(inclusivity_score)

                pred_peaks_tensor = pred_peaks_tensor.squeeze(0)
                if sample_idx == 0:
                    previous_pred_peaks = pred_peaks_tensor
                else:
                    all_consistency.append(torch.argmin(torch.norm(pred_peaks_tensor.unsqueeze(1) - previous_pred_peaks.unsqueeze(0), dim=2, p=None), 1).cpu().tolist())
                    previous_pred_peaks = pred_peaks_tensor

            logger.info("Test Result: inclusivity = {:.2f}".format(np.mean(all_inclusivity)))
            logger.info("Test Result: coverage = {:.2f}".format(np.mean(all_coverage)))

            temporal_consistency = []
            consistency_list = np.asarray(all_consistency)
            for j in range(consistency_list.shape[1]):
                temporal_consistency.append(np.sum(consistency_list[:, j] == j) / len(consistency_list) * 100)
            logger.info(f'Average temporal_consistency for whole test set: {np.mean(temporal_consistency)}')

        logger.info("Syncing ...")
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class KeypointTesterCAPE(TesterBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.metrics = KeypointMetrics()

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        save_path_pcd = os.path.join(self.cfg.save_path, "pcds")
        os.makedirs(save_path_pcd, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        comm.synchronize()
        all_pred_peaks_list = []
        all_coords = []

        for idx, input_dict in enumerate(self.test_loader):
            end = time.time()
            input_dict = input_dict[0]
            name = input_dict["name"]
            all_coords.append(input_dict["coord"].cpu().numpy())

            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.model(input_dict)

            all_pred_peaks = output_dict["all_pred_peaks"]
            all_pred_peaks = torch.stack(all_pred_peaks, dim=0)
            all_pred_peaks_list.append(all_pred_peaks)

            pred_peaks_tensor = all_pred_peaks[0].cpu().numpy()
            coord = input_dict["coord"]
            tree = cKDTree(coord.cpu())
            dists, indices = tree.query(pred_peaks_tensor, k=1)
            pred_peaks_surface = coord[indices]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord.cpu().numpy())

            safe_name = name.replace("/", "_").replace(".", "_")
            o3d.io.write_point_cloud(os.path.join(save_path_pcd, f"{safe_name}.pcd"), pcd)
            np.save(os.path.join(save_path, f"{safe_name}.npy"), pred_peaks_surface.cpu().numpy())

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                    name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                )
            )

        if comm.is_main_process():
            all_coverage = []
            all_inclusivity = []
            all_consistency = []

            for sample_idx, (pred_peaks, coords) in enumerate(
                    zip(all_pred_peaks_list, all_coords)):
                pred_peaks_np = np.stack([p.cpu().numpy() for p in pred_peaks])

                pred_peaks_tensor = torch.from_numpy(pred_peaks_np).float()
                coords_tensor = torch.from_numpy(coords).float().unsqueeze(0)
                coverage_score = self.metrics.coverage(pred_peaks_tensor, coords_tensor)
                inclusivity_score = self.metrics.inclusivity(pred_peaks_tensor.squeeze(0), coords_tensor.squeeze(0), threshold=0.05)
                all_coverage.append(coverage_score)
                all_inclusivity.append(inclusivity_score)

                pred_peaks_tensor = pred_peaks_tensor.squeeze(0)
                if sample_idx == 0:
                    previous_pred_peaks = pred_peaks_tensor
                else:
                    all_consistency.append(torch.argmin(torch.norm(pred_peaks_tensor.unsqueeze(1) - previous_pred_peaks.unsqueeze(0), dim=2, p=None), 1).cpu().tolist())
                    previous_pred_peaks = pred_peaks_tensor

            logger.info("Test Result: inclusivity = {:.2f}".format(np.mean(all_inclusivity)))
            logger.info("Test Result: coverage = {:.2f}".format(np.mean(all_coverage)))

            consistency_list = np.asarray(all_consistency)
            temporal_consistency = []
            for j in range(consistency_list.shape[1]):
                temporal_consistency.append(np.sum(consistency_list[:, j] == j) / len(consistency_list) * 100)
            logger.info(f'Average temporal_consistency for whole test set: {np.mean(temporal_consistency)}')

        logger.info("Syncing ...")
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch
