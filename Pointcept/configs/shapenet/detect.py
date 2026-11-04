_base_ = ["../_base_/default_runtime.py"]

batch_size = 8
empty_cache = False
enable_amp = True

num_keypoints = 10
num_coordinates = 10

# model settings
model = dict(
    type="DefaultKeypointerMatch",
    num_keypoints=num_keypoints,
    backbone=dict(
        type="PT-v2m2",
        in_channels=3,
        num_classes=num_coordinates,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.15, 0.375, 0.9375),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="map",
    ),
    criteria=[dict(type=f'KeypointLossMatch', loss_weight=1.0)],
)

# scheduler settings
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.001)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)


# dataset settings
train_datset = "ShapeNetDataset"
test_datset = "KeyPointNetDataset"
train_root = "data/shapenet"
test_root = "data/keypointnet"
shapenet_label = dict(
    airplane=0, bathtub=3, bed=4, bottle=6, cap=12, car=13,
    chair=14, guitar=27, helmet=28, knife=30, laptop=32, motorcycle=37,
    mug=38, skateboard=47, table=18, vessel=50,
)
category = 'airplane'
subclasses = (shapenet_label[category],)
knn = 20
pseudo = num_keypoints
use_fps = True
n_farthest = 100
ignore_index = -1


data = dict(
    ignore_index=ignore_index,
    num_classes=num_coordinates,
    train=dict(
        type=train_datset,
        data_root=train_root,
        split="train",
        save_record=True,
        normalize=True,
        include_label=True,
        subclasses=subclasses,
        knn=knn,
        pseudo=pseudo,
        use_fps=use_fps,
        n_farthest=n_farthest,
        transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "kp_index", "name", "pseudo_index"),
                feat_keys=("coord",),
                offset_keys_dict=dict(offset="coord", offset_pseudo="pseudo_index"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=test_datset,
        data_root=test_root,
        split="val",
        save_record=True,
        category=category,
        knn=knn,
        pseudo=pseudo,
        use_fps=use_fps,
        n_farthest=n_farthest,
        transform=[
            dict(type="MinMaxNormalize"),
            # dict(type="RandomJitter", sigma=0.20),
            # dict(type="FarthestPointSampling", down_points=64),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "kp_index", "name", "coord_original", "pseudo_index", "keypoints", "semantic_id"),
                feat_keys=("coord",),
                offset_keys_dict=dict(offset="coord", offset_pseudo="pseudo_index")
            ),

        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=test_datset,
        data_root=test_root,
        split="all",
        save_record=True,
        category=category,
        knn=knn,
        pseudo=pseudo,
        use_fps=use_fps,
        n_farthest=n_farthest,
        transform=[
            dict(type="MinMaxNormalize"),
            # dict(type="RandomJitter", sigma=0.20),
            # dict(type="FarthestPointSampling", down_points=64),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "kp_index", "name", "coord_original", "keypoints", "semantic_id"),
                feat_keys=("coord",),
                offset_keys_dict=dict(offset="coord"),
            ),
        ],
        test_mode=False,
    ),
)

# hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type=f'KeypointEvaluatorMatch'),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# tester
test = dict(type=f'KeypointTesterMatch', verbose=True)