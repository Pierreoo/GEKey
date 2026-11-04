import torch
import numpy as np


class KeypointMetrics:
    @staticmethod
    def compute_iou(pred, gt, threshold):
        npos = 0
        fp_sum = 0
        fn_sum = 0

        for ground_truths, kpcd in zip(gt, pred):
            kpcd_e = np.expand_dims(kpcd, 1)
            gt_e = np.expand_dims(ground_truths, 0)
            dist = np.sqrt(np.sum(np.square(kpcd_e - gt_e), -1))
            npos += len(np.min(dist, -2))
            fp_sum += np.count_nonzero(np.min(dist, -1) > threshold)
            fn_sum += np.count_nonzero(np.min(dist, -2) > threshold)

        return (npos - fn_sum) / (npos + fp_sum)

    @staticmethod
    def coverage(kp, pc):
        device = kp.device
        val_max_pc, _ = torch.max(pc, 1)
        val_min_pc, _ = torch.min(pc, 1)
        dim_pc = val_max_pc - val_min_pc
        val_max_kp, _ = torch.max(kp, 1)
        val_min_kp, _ = torch.min(kp, 1)
        dim_kp = val_max_kp - val_min_kp

        temp = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
        dis_kp = torch.cdist(temp, dim_kp).squeeze()
        dis_pc = torch.cdist(temp, dim_pc).squeeze()

        overlapping = 1 - torch.abs(dis_pc - dis_kp) / dis_pc
        overlapping[overlapping < 0] = 0
        return torch.mean(overlapping) * 100

    @staticmethod
    def inclusivity(kp, pc, threshold=0.03):
        loss = torch.cat([torch.squeeze(
            torch.norm(pc.unsqueeze(1) - kp.unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[0])], dim=0)
        return torch.tensor((len(loss[loss < threshold]) / len(loss)) * 100)

    @staticmethod
    def DAS_unsupervised(kpts):
        ref = np.expand_dims(kpts[0], 1)
        ref_index = [i for i in range(len(ref))]
        predictions = kpts[1:]

        positive = 0
        for pred in predictions:
            pred_ = np.expand_dims(pred, 0)
            dist = np.sqrt(np.sum(np.square(ref - pred_), -1))
            index_min_dist = np.argmin(dist, -1)
            positive += np.count_nonzero(index_min_dist == ref_index)

        positive = (positive / (len(predictions) * len(ref))) * 100
        return positive

    @staticmethod
    def fwd_alignment_scores(preds):
        acc = []
        for pa in preds:
            for pb in preds:
                acc.append(np.mean(pa == pb))
        return np.mean(acc)

    @staticmethod
    def bwd_alignment_scores(preds):
        q = []
        for semarr in preds.values():
            semarr = np.array(semarr, dtype=np.int16)
            q.append(np.mean(np.expand_dims(semarr, -1) == np.expand_dims(semarr, 0)))
        return np.mean(q)