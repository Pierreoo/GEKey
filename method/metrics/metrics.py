import torch
import numpy as np
import open3d as o3d
import seaborn as sns
import collections


def pc_to_pcd(pc, color=7):
    palette_PC = sns.color_palette()
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    pcd.translate(pc[0])
    pcd.paint_uniform_color(palette_PC[7])

    ''' Add points in the original point cloud'''
    for i in range(len(pc)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)  ## 0.005
        point.translate(pc[i])
        point.paint_uniform_color(palette_PC[7])
        pcd += point

    return pcd


def kp_to_pcd(kp):
    palette = sns.color_palette("bright")
    palette_dark = sns.color_palette("dark")
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)
    pcd.translate(kp[0])
    pcd.paint_uniform_color(palette[0])

    for i in range(1, len(kp)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)  # ablation: 0.035, figures: 0.050
        point.translate(kp[i])
        if i == 7:
            point.paint_uniform_color(palette_dark[7])
        else:
            point.paint_uniform_color(palette[i])
        pcd += point
    return pcd



def coverage(kp, pc):
    '''

    Parameters: 3D Coverage loss
    ----------
    kp: Estimated key-points [BxNx3]
    pc: Point cloud [Bx2048x3]

    Returns: Int value -> IoU b/w kp and pc
    -------

    '''
    device = kp.device
    val_max_pc, _ = torch.max(pc, 1)  # Bx3
    val_min_pc, _ = torch.min(pc, 1)  # Bx3
    dim_pc = val_max_pc - val_min_pc  # Bx3
    val_max_kp, _ = torch.max(kp, 1)  # Bx3
    val_min_kp, _ = torch.min(kp, 1)  # Bx3
    dim_kp = val_max_kp - val_min_kp  # Bx3

    ''' % coverage of kp over pc'''
    temp = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
    dis_kp = torch.cdist(temp, dim_kp).squeeze()  # distance of kp (BB) from origin
    dis_pc = torch.cdist(temp, dim_pc).squeeze()  # distance of PC (BB) from origin

    overlapping = 1 - torch.abs(dis_pc - dis_kp) / dis_pc
    overlapping[overlapping < 0] = 0
    return torch.mean(overlapping) * 100


def inclusivity(kp, pc, threshold=0.03):
    '''
    Parameters
    ----------
    pc      	Input point cloud
    kp      	Estimated key-points
    threshold  	threshold value

    Returns Shape loss -> how far the key-points are estimated from the input point cloud
    -------

    '''
    loss = torch.cat([torch.squeeze(
        torch.norm(pc.unsqueeze(1) - kp.unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[0])], dim=0)

    return torch.tensor((len(loss[loss < threshold]) / len(loss)) * 100)
    # return loss  # percentage of points closest to the surface


def DAS_unsupervised(kpts):
    '''
        Compute the DAS values:
        input: kpts (list of keypints for all the samples of the text set) i.e. [samples, keypoints, channel] [4920, 10, 3]
        Ref: first keypoint from the sets of the keypoints  [1,2,3,4,5]
        Pred: select keypoints for each sample one by one (other than the reference)

        Procedure:
        For every point in Ref, compute its 1NN from the Pred.
        positive += 1 if the indexes of both the points (Ref[i], 1NN(Pred, Ref[i])) are same
        Compute the ratio: total_positive / total KP * 100

    Returns
    -------

        Ratio: the percentage of corresponding keypoints

    '''

    ref = np.expand_dims(kpts[0], 1)  # k1 x 1 x 3
    ref_index = [i for i in range(len(ref))]
    predictions = kpts[1:]

    positive = 0
    for pred in predictions:
        pred_ = np.expand_dims(pred, 0)  # 1 x k2 x 3
        dist = np.sqrt(np.sum(np.square(ref - pred_), -1))  # k1 x k2
        index_min_dist = np.argmin(dist, -1)  # Index of KNN point of pred [K1, 1]
        positive += np.count_nonzero(index_min_dist == ref_index)

    positive = (positive / (len(predictions) * len(ref))) * 100  # [positive / total keypoints (samples*No_of_kp)] * 100

    return positive


def fwd_alignment_scores(preds):
    acc = []
    for pa in preds:
        for pb in preds:
            acc.append(np.mean(pa == pb))
    return np.mean(acc)


def bwd_alignment_scores(preds):
    q = []
    for semarr in preds.values():
        semarr = np.array(semarr, dtype=np.int16)
        q.append(np.mean(np.expand_dims(semarr, -1) == np.expand_dims(semarr, 0)))
    return np.mean(q)


def mIoU(gt_all, pred_all, threshold=0.1):
    npos = 0
    fp_sum = 0
    fn_sum = 0
    for ground_truths, kpcd in zip(gt_all, pred_all):
        kpcd_e = np.expand_dims(kpcd, 1)
        gt_e = np.expand_dims(ground_truths, 0)
        dist = np.sqrt(np.sum(np.square(kpcd_e - gt_e), -1))
        npos += len(np.min(dist, -2))
        fp_sum += np.count_nonzero(np.min(dist, -1) > threshold)
        fn_sum += np.count_nonzero(np.min(dist, -2) > threshold)
    return ((npos - fn_sum) / (npos + fp_sum)) * 100
