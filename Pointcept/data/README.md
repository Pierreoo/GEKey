# Data Layout

Place each dataset under `Pointcept/data/` as below.
Download links are listed under [Datasets](../../README.md#datasets).
Split and category files ship with the repository, and preprocessed point clouds, graphs and pseudo-label records are created in each dataset folder.

```
Pointcept/data/
│
├── shapenet/
│   └── train/               # train*.h5
├── keypointnet/
│   ├── annotations/         # <category>.json
│   └── pcds/                # <synset_id>/<model_id>.pcd
├── cape/
│   └── data/
│       ├── 00032/           # <sequence>/<sequence>.<frame>.npz
│       └── 00096/           # <sequence>/<sequence>.<frame>.npz
├── itop/                    # ITOP_side_train_{point_cloud,labels}.h5
├── deformingthings/
│   └── anime/               # <sequence>/<sequence>.anime
│
├── scanobjectnn/            # see scanobjectnn/README.md
├── smpl/                    # see smpl/README.md
└── dragon.obj               # xyzrgb_dragon.obj
```
