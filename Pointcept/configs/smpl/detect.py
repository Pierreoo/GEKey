_base_ = ["../_base_/default_runtime.py"]

batch_size = 32
evaluate = False
empty_cache = False
enable_amp = True

num_keypoints = 10
num_coordinates = 3

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
    final_div_factor=10000.0,
)


# dataset settings
dataset_type = "SMPLDataset"
data_root = "data/smpl"
knn = 20
pseudo = num_keypoints
use_fps = True
n_farthest = 100
ignore_index = -1

data = dict(
    ignore_index=ignore_index,
    num_classes=num_coordinates,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split="train",
        save_record=True,
        knn=knn,
        pseudo=pseudo,
        use_fps=use_fps,
        n_farthest=n_farthest,
        transform=[
            dict(type="MinMaxNormalize"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "name", "pseudo_index"),
                feat_keys=("coord",),
                offset_keys_dict=dict(offset="coord", offset_pseudo="pseudo_index"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        knn=knn,
        pseudo=pseudo,
        use_fps=use_fps,
        n_farthest=n_farthest,
        save_record=True,
        transform=[
            dict(type="MinMaxNormalize"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "name", "coord_original"),
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
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]


# tester
test = dict(type=f'KeypointTesterSMPL', verbose=True)