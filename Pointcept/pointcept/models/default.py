import torch.nn as nn
import torch.nn.functional as F

from .builder import MODELS, build_model
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point


class KeypointSelector(nn.Module):
    def __init__(self, num_keypoints=10, input_dim=3, hidden_dims=[512, 128, 32]):
        super().__init__()
        self.num_keypoints = num_keypoints

        layers = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.ReLU())
            last_dim = dim

        layers.append(nn.Linear(last_dim, num_keypoints))
        self.query_mlp = nn.Sequential(*layers)

    def forward(self, feats, coord):
        # weights = self.query_mlp(feats)
        weights = F.softmax(feats.transpose(0, 1), dim=-1)
        keypoints = weights @ coord
        return keypoints


@MODELS.register_module()
class DefaultKeypointerMatch(nn.Module):
    def __init__(self, backbone=None, num_keypoints=10, criteria=None, backbone_out_channels=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.selector = KeypointSelector(num_keypoints=num_keypoints, input_dim=3)
        self.head = (
            nn.Linear(backbone_out_channels, num_keypoints)
            if backbone_out_channels is not None
            else nn.Identity()
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        feat = point.feat if isinstance(point, Point) else point
        feat = self.head(feat)

        offset = nn.functional.pad(input_dict["offset"], (1, 0))

        # eval or test
        all_pred_peaks = []
        if not self.training and "pseudo_index" not in input_dict:
            for i in range(len(input_dict["offset"])):
                pred = feat[offset[i]:offset[i + 1]]
                coord = input_dict["coord"][offset[i]:offset[i + 1]]
                pred_peaks = self.selector(pred, coord)
                all_pred_peaks.append(pred_peaks)
            return dict(feat=feat, all_pred_peaks=all_pred_peaks)

        # train
        offset_pseudo = nn.functional.pad(input_dict["offset_pseudo"], (1, 0))
        all_peaks = []
        for i in range(len(input_dict["offset"])):
            pred = feat[offset[i]:offset[i + 1]]
            coord = input_dict["coord"][offset[i]:offset[i + 1]]
            pseudo_index = input_dict["pseudo_index"][offset_pseudo[i]:offset_pseudo[i + 1]]

            pred_peaks = self.selector(pred, coord)
            peaks = coord[pseudo_index]

            all_pred_peaks.append(pred_peaks)
            all_peaks.append(peaks)

        loss = self.criteria(all_pred_peaks, all_peaks)

        if self.training:
            return dict(loss=loss)
        else:
            return dict(loss=loss, all_pred_peaks=all_pred_peaks)
