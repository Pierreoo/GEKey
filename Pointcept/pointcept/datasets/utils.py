"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""


import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from scipy.spatial import distance
import scipy.sparse as ss
from scipy.sparse.csgraph import shortest_path


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if "offset" in batch.keys():
        if random.random() < mix_prob:
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=float)
    colors = np.array(data[:, -1], dtype=int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return np.hstack((pc, colors)).astype(np.float32)


def naive_read_smpl(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=float)
    return pc.astype(np.float32)


def _connect_components(adjacency, coord, repair_connections=1):
    n_components, labels = ss.csgraph.connected_components(adjacency, directed=False)
    if n_components == 1:
        return adjacency
    adjacency = adjacency.tolil()
    # Get unique pairs of disconnected components
    for i, j in np.transpose(np.triu_indices(n_components, k=1)):
        nodes_i = np.where(labels == i)[0]
        nodes_j = np.where(labels == j)[0]
        # Compute pairwise distances and get the indices of the closest n pairs
        distances = distance.cdist(coord[nodes_i], coord[nodes_j], 'euclidean')
        closest_pairs = np.unravel_index(np.argsort(distances, axis=None)[:repair_connections], distances.shape)
        # Map the closest pairs to original node indices
        closest_i = nodes_i[closest_pairs[0]]
        closest_j = nodes_j[closest_pairs[1]]
        # Add the distances as weights to the adjacency matrix in both directions
        adjacency[closest_i, closest_j] = distances[closest_pairs]
        adjacency[closest_j, closest_i] = distances[closest_pairs]
    return adjacency.tocsr()
