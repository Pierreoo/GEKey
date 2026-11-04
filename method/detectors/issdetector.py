import numpy as np
import open3d as o3d


class ISSDetector:
    def __init__(self, cloud, coords, num_keys=10, salient_radius=0.05, non_max_radius=0.05, gamma_21=0.5, gamma_32=0.5):
        self.cloud = cloud
        self.coords = coords
        self.num_keys = num_keys
        self.salient_radius = salient_radius
        self.non_max_radius = non_max_radius
        self.gamma_21 = gamma_21
        self.gamma_32 = gamma_32

    def farthest_point_sampling(self, candidates, k, initial_nodes=None):
        if not candidates or k <= 0:
            return []

        candidates = list(candidates)
        initial_nodes = initial_nodes or []
        distances = {i: np.inf for i in candidates}

        for i in candidates:
            for j in initial_nodes:
                d = np.linalg.norm(self.coords[i] - self.coords[j])
                if d < distances[i]:
                    distances[i] = d

        selected = []
        for _ in range(min(k, len(candidates))):
            idx = max(candidates, key=lambda i: distances[i])
            selected.append(idx)
            candidates.remove(idx)

            for i in candidates:
                d = np.linalg.norm(self.coords[i] - self.coords[idx])
                if d < distances[i]:
                    distances[i] = d

        return selected

    def detect(self):
        iss_pcd = o3d.geometry.keypoint.compute_iss_keypoints(
            self.cloud,
            salient_radius=self.salient_radius,
            non_max_radius=self.non_max_radius,
            gamma_21=self.gamma_21,
            gamma_32=self.gamma_32,
        )
        iss_points = np.asarray(iss_pcd.points)

        # Match ISS points back to original point cloud indices
        iss_indices = []
        for p in iss_points:
            dists = np.linalg.norm(self.coords - p, axis=1)
            iss_indices.append(int(np.argmin(dists)))

        if len(iss_indices) > self.num_keys:
            selected = self.farthest_point_sampling(iss_indices, self.num_keys)
            return selected, []
        elif len(iss_indices) == self.num_keys:
            return iss_indices, []
        else:
            remaining_k = self.num_keys - len(iss_indices)
            remaining_candidates = set(range(len(self.coords))) - set(iss_indices)
            fps_additional = self.farthest_point_sampling(remaining_candidates, remaining_k, initial_nodes=iss_indices)
            return iss_indices, fps_additional

    def get_iss_spheres(self, radius=0.01):
        iss_indices, fps_additional = self.detect()
        all_indices = iss_indices + fps_additional

        spheres = []
        for idx in all_indices:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(self.coords[idx])
            spheres.append(sphere)
        return spheres