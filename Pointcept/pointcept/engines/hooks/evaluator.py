import numpy as np
import torch
import torch.distributed as dist
from collections import defaultdict
from scipy.spatial import cKDTree

import pointcept.utils.comm as comm
from pointcept.engines.metrics.keypoint import KeypointMetrics

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class KeypointEvaluatorMatch(HookBase):
    def __init__(self):
        super().__init__()
        self.metrics = KeypointMetrics()
        self.additional_metrics = False

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        losses = []
        all_names = []
        all_pred_peaks_list = []
        all_gt_peaks_list = []
        all_coords = []
        all_coords_original = []
        all_gt_keypoints = []
        all_sematics = []

        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]
            losses.append(loss.item())

            all_pred_peaks = output_dict["all_pred_peaks"]
            all_pred_peaks = torch.stack(all_pred_peaks, dim=0)

            label = input_dict["kp_index"][0].squeeze()
            gt_indices = torch.where(label == 1)[0]
            all_gt_peaks = input_dict["coord_original"][gt_indices].unsqueeze(0)

            all_pred_peaks_list.append(all_pred_peaks)
            all_gt_peaks_list.append(all_gt_peaks)

            name = input_dict["name"]
            all_names.append(name[0])
            all_coords.append(input_dict["coord"].cpu().numpy())
            all_coords_original.append(input_dict["coord_original"].cpu().numpy())
            all_gt_keypoints.append(input_dict["keypoints"])
            all_sematics.append(input_dict["semantic_id"])

            info = f"Val: [{i + 1}/{len(self.trainer.val_loader)}] Loss {loss.item():.4f}"
            self.trainer.logger.info(info)

        miou_vals = []
        for i in range(11):
            dist_thresh = 0.01 * i
            total_iou = 0.0

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

                if self.additional_metrics:
                    pred_peaks_tensor = torch.from_numpy(pred_peaks_np).float()
                    coords_tensor = torch.from_numpy(coords).float().unsqueeze(0)
                    coverage_score = self.metrics.coverage(pred_peaks_tensor, coords_tensor)
                    inclusivity_score = self.metrics.inclusivity(pred_peaks_tensor.squeeze(0), coords_tensor.squeeze(0), threshold=0.10)
                    all_coverage.append(coverage_score)
                    all_inclusivity.append(inclusivity_score)

                all_pred_peaks_surface.append(pred_peaks_surface.squeeze(0))
                all_gt_peaks_np.append(gt_peaks_np.squeeze(0))

                iou = self.metrics.compute_iou(pred_peaks_surface, gt_peaks_np, dist_thresh)
                total_iou += iou

                if self.additional_metrics:
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

            miou_val = self.metrics.compute_iou(all_pred_peaks_surface, all_gt_peaks_np, dist_thresh)

            self.trainer.logger.info(f"Val Result @ dist {dist_thresh:.2f}: mIoU = {miou_val:.4f}")
            miou_vals.append(miou_val)

        if comm.get_world_size() > 1:
            miou_vals = torch.tensor(miou_vals, device="cuda")
            dist.all_reduce(miou_vals)
            miou_vals = (miou_vals / comm.get_world_size()).tolist()

        if self.additional_metrics:
            combined_preds_forward = np.array(combined_preds_forward)
            das_forward = self.metrics.fwd_alignment_scores(combined_preds_forward)
            das_backward = self.metrics.bwd_alignment_scores(combined_preds_backward)
            das_sm = ((das_backward + das_forward) / 2) * 100

        all_pred_peaks_tensor = torch.cat(all_pred_peaks_list, dim=0)
        all_pred_peaks_np_list = [p.cpu().numpy() for p in all_pred_peaks_tensor]
        das_sc3k = self.metrics.DAS_unsupervised(all_pred_peaks_np_list)

        if self.additional_metrics:
            self.trainer.logger.info("Val Result: inclusivity = {:.2f}".format(np.mean(all_inclusivity)))
            self.trainer.logger.info("Val Result: coverage = {:.2f}".format(np.mean(all_coverage)))
            self.trainer.logger.info("Val Result: DAS_SM = {:.2f}".format(das_sm))

        self.trainer.logger.info(f"Val Result: DAS = {das_sc3k:.2f}")
        self.trainer.storage.put_scalar("DAS", das_sc3k)

        loss_avg = sum(losses) / len(losses)
        for i in range(11):
            thresh = i * 0.01
            self.trainer.storage.put_scalar(f'mIoU-{thresh:.2f}', miou_vals[i])

        self.trainer.storage.put_scalar("val_loss", loss_avg)

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            for i in range(11):
                thresh = i * 0.01
                self.trainer.writer.add_scalar(f"val/mIoU-{thresh:.2f}", self.trainer.storage.history(f'mIoU-{thresh:.2f}').avg, current_epoch)
            self.trainer.writer.add_scalar("val/DAS", das_sc3k, current_epoch)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = self.trainer.storage.history('mIoU-0.10').avg
        self.trainer.comm_info["current_metric_name"] = "mIoU-0.10"

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU-0.10", self.trainer.best_metric_value)
        )
