import torch
import spconv.pytorch as spconv

try:
    import ocnn
except ImportError:
    ocnn = None
from addict import Dict

from pointcept.models.utils import offset2batch, batch2offset


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    def octreetization(self, depth=None, full_depth=None):
        """
        Point Cloud Octreelization

        Generate octree with OCNN
        relay on ["grid_coord", "batch", "feat"]
        """
        assert (
            ocnn is not None
        ), "Please follow install ocnn."
        assert {"grid_coord", "feat", "batch"}.issubset(self.keys())
        # add 1 to make grid space support shift order
        if depth is None:
            if "depth" in self.keys():
                depth = self.depth
            else:
                depth = int(self.grid_coord.max() + 1).bit_length()
        if full_depth is None:
            full_depth = 2
        self["depth"] = depth
        assert depth <= 16  # maximum in ocnn

        # [0, 2**depth] -> [0, 2] -> [-1, 1]
        coord = self.grid_coord / 2 ** (self.depth - 1) - 1.0
        point = ocnn.octree.Points(
            points=coord,
            features=self.feat,
            batch_id=self.batch.unsqueeze(-1),
            batch_size=self.batch[-1] + 1,
        )
        octree = ocnn.octree.Octree(
            depth=depth,
            full_depth=full_depth,
            batch_size=self.batch[-1] + 1,
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()
        self["octree"] = octree
