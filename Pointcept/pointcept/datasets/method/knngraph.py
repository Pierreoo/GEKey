import numpy as np
import open3d as o3d

from scipy.spatial import distance
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components, dijkstra


class CSRAdjacency:
    def __init__(self, csr):
        self.csr = csr.tocsr()

    @property
    def nodes(self):
        return range(self.csr.shape[0])

    def neighbors(self, i):
        return self.csr.indices[self.csr.indptr[i]:self.csr.indptr[i + 1]]

    def edges(self):
        coo = self.csr.tocoo()
        return zip(coo.row.tolist(), coo.col.tolist())

    def compute_geodesics_full(self):
        """All-pairs geodesic distances (single scipy csgraph Dijkstra call)."""
        geo_dists = dijkstra(self.csr, directed=False).astype(np.float64)
        np.nan_to_num(geo_dists, posinf=0.0, copy=False)
        return geo_dists


class KnnGraph:
    def __init__(self, coord, k=20, use_fps=False, n_farthest=512):
        self.coord = coord
        self.k = k
        self.n_farthest = n_farthest
        self.graph = self.build_knn_graph()
        self.graph = self.graph.maximum(self.graph.T)
        self.graph = self.connect_components(self.graph, self.coord)
        self.G = CSRAdjacency(self.graph)
        self.geo_dists = self.compute_geodesics_map(use_fps=use_fps, n_farthest=n_farthest)

    def build_knn_graph(self):
        """Build graph from k-nearest neighbors"""
        return kneighbors_graph(self.coord, self.k, mode='distance', include_self=False)

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

    def compute_geodesics_map(self, use_fps=False, n_farthest=100):
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
            rows = dijkstra(self.graph, directed=False, indices=source_indices[start:start + chunk])
            np.nan_to_num(rows, posinf=0.0, copy=False)
            dists = np.maximum(dists, rows.max(axis=0))

        return dists

    def get_graph_and_distances(self):
        """Get the CSR graph view and the eccentricity map"""
        return self.G, self.geo_dists
