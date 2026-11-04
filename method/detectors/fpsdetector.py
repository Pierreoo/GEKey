import numpy as np
import open3d as o3d


class FPSDetector:
    def __init__(self, coords, num_samples=20):
        self.coords = coords
        self.num_samples = num_samples

    def detect_fps_points(self):
        coords = np.asarray(self.coords)
        N = len(coords)
        selected = []
        dists = np.full(N, np.inf)

        initial_index = 0
        selected.append(initial_index)

        for _ in range(1, self.num_samples):
            last_selected = selected[-1]
            last_coord = coords[last_selected]

            dist_to_last = np.linalg.norm(coords - last_coord, axis=1)
            dists = np.minimum(dists, dist_to_last)

            next_index = np.argmax(dists)
            selected.append(next_index)

        return selected

    def get_fps_spheres(self, radius=0.015):
        fps_indices = self.detect_fps_points()
        spheres = []
        for idx in fps_indices:
            sphere = o3d.geometry.TriangleMesh.create_tetrahedron(radius=radius)
            sphere.translate(self.coords[idx])
            sphere.paint_uniform_color([0.0, 0.5, 0.0])
            spheres.append(sphere)
        print(f"number of fps points = {len(fps_indices)}")
        return spheres
