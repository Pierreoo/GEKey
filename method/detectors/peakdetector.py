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
            return selected_peaks, []
        elif len(significant_peaks) == self.num_peaks:
            selected_peaks = list(significant_peaks)
            return selected_peaks, []
        else:
            selected_peaks = list(significant_peaks)
            remaining_k = self.num_peaks - len(selected_peaks)
            remaining_candidates = set(self.G.nodes) - set(selected_peaks)
            fps_additional = self.farthest_point_sampling(remaining_candidates, remaining_k, initial_nodes=selected_peaks)
            return selected_peaks, fps_additional

    def compute_heights(self, normalize=True):
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

        if normalize:
            min_h = heights.min()
            max_h = heights.max()
            denom = max_h - min_h if max_h != min_h else 1.0
            heights = (heights - min_h) / denom

        return heights

    def get_peak_spheres(self, radius=0.010, points=False, verbose=True):
        peaks, fps_peaks = self.detect_peaks()
        if points:
            peaks_coords = self.coords[peaks]
            fps_coords = self.coords[fps_peaks]
            all_keypoints = np.vstack([peaks_coords, fps_coords])
            if verbose:
                print(f"Peaks ({len(peaks)}) | FPS Peaks ({len(fps_peaks)})")
            return all_keypoints
        else:
            spheres = []
            for node in peaks:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(self.coords[node])
                # sphere.paint_uniform_color([0.45, 0.65, 0.40])
                sphere.paint_uniform_color([0.45, 0.65, 0.40])
                spheres.append(sphere)
            for node in fps_peaks:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(self.coords[node])
                sphere.paint_uniform_color([1.0, 0.1, 0.1])
                spheres.append(sphere)
            if verbose:
                print(f"Peaks ({len(peaks)}) | FPS Peaks ({len(fps_peaks)})")
            return spheres

    def visualize_heights(self, heights, use_spheres=True):
        edges = np.array([[u, v] for u, v in self.G.edges()])
        avg_heights = np.array([(heights[u] + heights[v]) / 2 for u, v in self.G.edges()])

        colors = [plt.cm.plasma(h)[:3] for h in avg_heights]

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(self.coords)
        lines.lines = o3d.utility.Vector2iVector(edges)
        lines.colors = o3d.utility.Vector3dVector(colors)

        geometries = [lines]

        if use_spheres:
            sphere_radius = 0.0015
            for point in self.coords:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                sphere.translate(point)
                sphere.paint_uniform_color([0.2, 0.8, 0.2])
                sphere.compute_vertex_normals()
                geometries.append(sphere)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.coords)
            pcd.paint_uniform_color([0.2, 0.8, 0.2])
            geometries.append(pcd)

        o3d.visualization.draw_geometries(geometries)