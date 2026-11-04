import numpy as np
import open3d as o3d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import Delaunay, distance
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse import lil_matrix

from method.graphs.knngraph import CSRAdjacency


class DelaunayGraph:
    def __init__(self, coord, use_fps=False, n_farthest=512):
        self.coord = coord
        self.n_farthest = n_farthest
        self.graph = self.build_delaunay_graph()
        self.graph = self.connect_components(self.graph, self.coord)
        self.G = CSRAdjacency(self.graph)
        self.geo_dists = self.compute_geodesics_map(use_fps=use_fps, n_farthest=n_farthest)

    def build_delaunay_graph(self):
        """Build graph from Delaunay triangulation"""
        tri = Delaunay(self.coord)

        n_points = len(self.coord)
        adjacency = lil_matrix((n_points, n_points))

        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    idx_i = simplex[i]
                    idx_j = simplex[j]

                    dist = np.linalg.norm(self.coord[idx_i] - self.coord[idx_j])

                    adjacency[idx_i, idx_j] = dist
                    adjacency[idx_j, idx_i] = dist

        return adjacency.tocsr()

    def connect_components(self, adjacency, coord, repair_connections=1):
        """Connect disconnected components in the graph"""
        n_components, labels = connected_components(adjacency, directed=False)
        if n_components == 1:
            return adjacency

        adjacency = adjacency.tolil()
        for i, j in np.transpose(np.triu_indices(n_components, k=1)):
            nodes_i = np.where(labels == i)[0]
            nodes_j = np.where(labels == j)[0]
            distances = distance.cdist(coord[nodes_i], coord[nodes_j], 'euclidean')
            closest_pairs = np.unravel_index(
                np.argsort(distances, axis=None)[:repair_connections],
                distances.shape
            )
            closest_i = nodes_i[closest_pairs[0]]
            closest_j = nodes_j[closest_pairs[1]]
            adjacency[closest_i, closest_j] = distances[closest_pairs]
            adjacency[closest_j, closest_i] = distances[closest_pairs]

        return adjacency.tocsr()

    def compute_geodesics_map(self, use_fps=False, n_farthest=100, normalize=True):
        """Per-point eccentricity map = max geodesic distance to any source, via scipy csgraph
        Dijkstra. Sources are FPS samples (use_fps) or all points. Chunked so the all-points path
        never materializes the full N x N distance matrix."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coord)

        if use_fps:
            sampled_pcd = pcd.farthest_point_down_sample(n_farthest)
            sampled_points = np.asarray(sampled_pcd.points)
            coord_tree = o3d.geometry.KDTreeFlann(pcd)
            source_indices = []
            for pt in sampled_points:
                [_, idx, _] = coord_tree.search_knn_vector_3d(pt, 1)
                source_indices.append(idx[0])
            source_indices = np.array(source_indices)
        else:
            source_indices = np.arange(len(self.coord))

        num_points = len(self.coord)
        dists = np.zeros(num_points, dtype=np.float64)
        chunk = 256
        for start in range(0, len(source_indices), chunk):
            rows = dijkstra(self.graph, directed=False,
                            indices=source_indices[start:start + chunk])
            np.nan_to_num(rows, posinf=0.0, copy=False)
            dists = np.maximum(dists, rows.max(axis=0))

        if normalize:
            min_val = np.min(dists)
            max_val = np.max(dists)
            denom = max_val - min_val if max_val != min_val else 1.0
            dists = (dists - min_val) / denom

        return dists

    def visualize(self, use_spheres=True):
        """Visualize the Delaunay graph with colored points"""
        rows, cols = self.graph.nonzero()
        edges = np.array([[i, j] for i, j in zip(rows, cols)])

        point_colors = np.array([plt.cm.plasma(d)[:3] for d in self.geo_dists])

        edge_colors = [plt.cm.plasma((self.geo_dists[i] + self.geo_dists[j]) / 2)[:3] for i, j in edges]

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(self.coord)
        lines.lines = o3d.utility.Vector2iVector(edges)
        lines.colors = o3d.utility.Vector3dVector(edge_colors)

        geometries = [lines]

        if use_spheres:
            small_radius = 0.01
            for idx, point in enumerate(self.coord):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=small_radius)
                sphere.translate(point)
                sphere.paint_uniform_color(point_colors[idx])
                sphere.compute_vertex_normals()
                geometries.append(sphere)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.coord)
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
            geometries.append(pcd)

        o3d.visualization.draw_geometries(geometries)

    def show_colorbar(self, colormap='plasma', vmin=0, vmax=1, fontsize=16):
        """Display a colorbar for the visualization"""
        fig, ax = plt.subplots(figsize=(1, 6))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax, orientation='vertical')
        ticks = [vmin, vmax]
        tick_labels = [f"{vmin:.0f}", f"{vmax:.0f}"]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=fontsize)
        plt.tight_layout()
        plt.show()

    def get_graph_and_distances(self):
        """Get the CSR graph view and the eccentricity map"""
        return self.G, self.geo_dists