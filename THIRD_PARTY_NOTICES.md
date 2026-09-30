# Third-party notices

My own code in this repository is released under the MIT License (`LICENSE`). The files listed below contain third-party
work and keep their original terms. **They are Apache-2.0, not MIT.**

## Code derived from third-party work (Apache License 2.0)

| Where | Origin |
|---|---|
| Cell 2 of every notebook in `notebooks/`, and `lineage_models/b56/cloud_baseline.py` | The public Kaggle notebook [*Biohub 0.947 LB, runnable with public datasets*](https://www.kaggle.com/code/zhincez/biohub-0-947-lb-runnable-with-public-datasets) by **zhincez** (LÊ QUANG CẢNH), released under the Apache 2.0 open-source license. On Kaggle it is marked as copied from a notebook by **Reyhan Ksatria**, and its code records the source kernel `raykkretzschmar/biohub-bidirectional-primary-union13-diagnostic-v1`. The earlier B5 notebooks carried the attribution line *"Biohub competition authors, zhincez, Pilkwang, Reyhan Satria, nusrati and evgendvorkin"*, which I keep here. |

The full license text is in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

**Statement of changes.** I modified this code by adding:

- an export of the pre-ILP candidate graph (`fullgraphs/*.geff`) and of the reference graphs;
- streaming of finished movies into my B5 lineage stages;
- runtime precision and batching patches;
- configuration changes;
- a 4-GPU switch used to regenerate training graphs.

`cloud_baseline.py` carries this notice in its header. The notebook cells carry it here, because editing them would change
the submitted bytes.

Cell 2 and `cloud_baseline.py` also contain short verbatim excerpts, about 1.6 KB, of pilkwang's inference script
`scripts/predict_unet_transformer.py`. They are inherited from the upstream notebook and used as runtime patch anchors. The
script itself is loaded at runtime from the support-pack dataset below, which is published under CC0.

## Referenced but not redistributed

| Component | Source | License |
|---|---|---|
| Temporal 3D U-Net detector, inference repository and offline wheels | Kaggle dataset [`pilkwang/biohub-tracking-support-pack-50ep-v1`](https://www.kaggle.com/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1) | CC0: Public Domain |
| Second detector seed | Kaggle dataset [`pilkwang/biohub-temporal-unet3d-seed314159-v1`](https://www.kaggle.com/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1) | CC0: Public Domain |
| DeepCenter centre prior | Kaggle dataset [`pilkwang/biohub-deepcenter-unet3d-center-prior-v1`](https://www.kaggle.com/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1) | CC0: Public Domain |
| Official metric (`tracking_cellmot`) | [royerlab/kaggle-cell-tracking-competition @ 075fc5f](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f5a52d11077f9dc2b074644618f26939e2) | see that repository; imported, never copied |
| Competition data | [Kaggle competition data](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data) | CC0 per the competition rules; only movie ids are included here (`results/splits/`) |

## Python packages

The notebooks and scripts use NumPy, SciPy, pandas, PyTorch, zarr, numcodecs, polars, ilpy, networkx, LightGBM (training
only) and the packages installed by the support pack. Each is under its own open-source license.
