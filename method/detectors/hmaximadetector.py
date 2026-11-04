import numpy as np
import open3d as o3d


class HMaximaDetector:
    def __init__(self, graph, geo_dists, coords, target_maxima=10, tolerance=0, max_iter=20):
        self.G = graph
        self.geo_dists = geo_dists
        self.coords = coords
        self.target_maxima = target_maxima
        self.tolerance = tolerance
        self.max_iter = max_iter

    def morphological_reconstruction(self, marker, mask):
        reconstructed = marker.copy()
        changed = True
        while changed:
            changed = False
            for node in self.G.nodes:
                neighbor_vals = [reconstructed[n] for n in self.G.neighbors(node)]
                max_neighbor_val = max(neighbor_vals, default=0)
                new_val = min(mask[node], max(reconstructed[node], max_neighbor_val))
                if new_val > reconstructed[node]:
                    reconstructed[node] = new_val
                    changed = True
        return reconstructed

    def find_regional_maxima(self, values):
        maxima = []
        for node in self.G.nodes:
            val = values[node]
            neighbor_vals = [values[n] for n in self.G.neighbors(node)]
            if all(val > nv for nv in neighbor_vals):
                maxima.append(node)
        return maxima

    def detect_hmaxima(self):
        h_low = 0.0
        h_high = np.max(self.geo_dists) / 2
        best_nodes = []
        best_h = None

        for _ in range(self.max_iter):
            h = (h_low + h_high) / 2
            lowered = {node: max(self.geo_dists[node] - h, 0) for node in self.G.nodes}
            reconstructed = self.morphological_reconstruction(lowered, self.geo_dists)
            maxima_nodes = self.find_regional_maxima(reconstructed)
            num_maxima = len(maxima_nodes)

            if abs(num_maxima - self.target_maxima) <= self.tolerance:
                best_nodes = maxima_nodes
                best_h = h
                break

            if num_maxima > self.target_maxima:
                h_low = h
            else:
                h_high = h

            if not best_nodes or abs(num_maxima - self.target_maxima) < abs(len(best_nodes) - self.target_maxima):
                best_nodes = maxima_nodes
                best_h = h

        additional_nodes = []
        if len(best_nodes) < self.target_maxima:
            num_needed = self.target_maxima - len(best_nodes)
            additional_nodes = self.farthest_point_sampling(best_nodes, num_needed, distance='euclidean')
            best_nodes += additional_nodes

        best_nodes = sorted(best_nodes, key=lambda node: self.geo_dists[node], reverse=True)
        if len(best_nodes) > self.target_maxima:
            best_nodes = best_nodes[:self.target_maxima]
            additional_nodes = []

        return best_nodes, best_h, additional_nodes

    def get_hmaxima_spheres(self, radius=0.02, verbose=False):
        nodes, h, additional_nodes = self.detect_hmaxima()
        spheres = []

        for node in nodes:
            sphere = o3d.geometry.TriangleMesh.create_tetrahedron(radius=radius)
            sphere.translate(self.coords[node])
            if node in additional_nodes:
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
            else:
                sphere.paint_uniform_color([0.0, 0.5, 0.0])
            spheres.append(sphere)
        if verbose:
            print(f"Selected h = {h:.6f}, h-maxima = {len(nodes) - len(additional_nodes)}, fps-added = {len(additional_nodes)}")
        return spheres

    def farthest_point_sampling(self, existing_indices, k, distance='geodesic'):
        if distance not in {'geodesic', 'euclidean'}:
            raise ValueError("distance must be either 'geodesic' or 'euclidean'")

        if distance == 'geodesic':
            all_indices = list(self.G.nodes)
        else:
            all_indices = list(range(len(self.coords)))

        remaining = list(set(all_indices) - set(existing_indices))
        if not remaining or k <= 0:
            return []

        geo_full = self.G.compute_geodesics_full() if distance == 'geodesic' else None

        selected = []

        if distance == 'geodesic':
            distances = {i: np.inf for i in all_indices}
            for i in all_indices:
                for j in existing_indices:
                    d = geo_full[i, j]
                    if d < distances[i]:
                        distances[i] = d
        else:
            distances = np.full(len(self.coords), np.inf)
            for i in remaining:
                for j in existing_indices:
                    d = np.linalg.norm(self.coords[i] - self.coords[j])
                    if d < distances[i]:
                        distances[i] = d

        for _ in range(min(k, len(remaining))):
            idx = max(remaining, key=lambda i: distances[i])
            selected.append(idx)
            remaining.remove(idx)

            if distance == 'geodesic':
                for i in remaining:
                    d = geo_full[i, idx]
                    if d < distances[i]:
                        distances[i] = d
            else:
                for i in remaining:
                    d = np.linalg.norm(self.coords[i] - self.coords[idx])
                    if d < distances[i]:
                        distances[i] = d

        return selected
