set -e
mkdir -p /kaggle/input/datasets/pilkwang /kaggle/input/datasets/shawsebastian /kaggle/input/competitions
ln -sfn /workspace/models/support /kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1
ln -sfn /workspace/models/deepcenter /kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1
ln -sfn /workspace/models/detector_seed2 /kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1
ln -sfn /workspace/models/b56 /kaggle/input/datasets/shawsebastian/biohub-b5-b6-track-video-models-20260923
ln -sfn /workspace/models/b34 /kaggle/input/datasets/shawsebastian/biohub-b3-b4-lineage-models-20260922
ln -sfn /workspace/data /kaggle/input/competitions/biohub-cell-tracking-during-development
ls -la /kaggle/input/datasets/pilkwang /kaggle/input/datasets/shawsebastian /kaggle/input/competitions
