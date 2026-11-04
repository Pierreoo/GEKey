import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


class PeakDetector:
    def __init__(self, graph, geodists, coords, num_peaks=10):
        self.G = graph
        self.geo_dists = geodists
        self.coords = coords
        self.num_peaks = num_peaks

    def farthest_point_sampling(self, candidates, k, initial_nodes=None, distance='euclidean'):
        if distance not in {'geodesic', 'euclidean'}:
            raise ValueError("distance must be either 'geodesic' or 'euclidean'")

        if not candidates or k <= 0:
            return []

        geo_full = self.G.compute_geodesics_full() if distance == 'geodesic' else None

        selected = []
        candidates = list(candidates)
        if initial_nodes is None:
            initial_nodes = []
        distances = {i: np.inf for i in candidates}

        if distance == 'geodesic':
            for i in candidates:
                for j in initial_nodes:
                    d = geo_full[i, j]
                    if d < distances[i]:
                        distances[i] = d
        else:
            for i in candidates:
                for j in initial_nodes:
                    d = np.linalg.norm(self.coords[i] - self.coords[j])
                    if d < distances[i]:
                        distances[i] = d

        for _ in range(min(k, len(candidates))):
            idx = max(candidates, key=lambda i: distances[i])
            selected.append(idx)
            candidates.remove(idx)

            if distance == 'geodesic':
                for i in candidates:
                    d = geo_full[i, idx]
                    if d < distances[i]:
                        distances[i] = d
            else:
                for i in candidates:
                    d = np.linalg.norm(self.coords[i] - self.coords[idx])
                    if d < distances[i]:
                        distances[i] = d
        return selected

    def detect_peaks(self):
        peaks_with_heights = []
        nodes_list = list(self.G.nodes)

        for node in self.G.nodes:
            node_dist = self.geo_dists[node]
            neighbors = list(self.G.neighbors(node))
            neighbor_dists = [self.geo_dists[n] for n in neighbors]
            max_neighbor_dist = max(neighbor_dists + [node_dist])
            if node_dist == max_neighbor_dist:
                peaks_with_heights.append((node, 0))

        peaks_with_heights.sort(key=lambda x: x[1], reverse=True)
        significant_peaks = [node for node, height in peaks_with_heights]
        peak_indices = [nodes_list.index(node) for node in significant_peaks]

        if len(significant_peaks) > self.num_peaks:
            selected_peaks = self.farthest_point_sampling(significant_peaks, self.num_peaks)
            selected_peaks = torch.tensor(selected_peaks)
        elif len(significant_peaks) == self.num_peaks:
            selected_peaks = torch.tensor(list(significant_peaks))
        else:
            selected_peaks = list(significant_peaks)
            remaining_k = self.num_peaks - len(selected_peaks)
            remaining_candidates = set(self.G.nodes) - set(selected_peaks)
            fps_additional = self.farthest_point_sampling(remaining_candidates, remaining_k, initial_nodes=selected_peaks)
            combined_peaks = selected_peaks + list(fps_additional)
            selected_peaks = torch.tensor(combined_peaks)

        return selected_peaks

    def compute_heights(self):
        num_nodes = len(self.coords)
        heights = torch.zeros(num_nodes)

        for node in self.G.nodes:
            node_dist = self.geo_dists[node]
            neighbors = list(self.G.neighbors(node))
            neighbor_dists = [self.geo_dists[n] for n in neighbors]

            max_neighbor_dist = max(neighbor_dists + [node_dist])

            if node_dist == max_neighbor_dist:
                height = 0.0
            else:
                height = max_neighbor_dist - node_dist

            heights[node] = height

        return heights
