# Lineage models (B1–B6)

This is the source of the learned lineage stages that sit between the public baseline and my P-stage. The files are copied
from the frozen model bundles, which is why flat module names are imported directly by `pstage/`: `refine_events` by the
`p_stage*.py` workers, and `cell_event` by `div_complete.py` and `dsr.py`.

Scripts that provisioned cloud machines or used credentials are omitted. A few on-machine job launchers remain as run and
assume `/workspace/biohub`: `start_*.py`, `add_b4_workers.py`, `enable_four_gpu_reference.py` and `stream_*training*.py`. The weights are not in this repository; they are in the public Kaggle dataset
[`shawsebastian/biohub-b5-b6-track-video-models-20260923`](https://www.kaggle.com/datasets/shawsebastian/biohub-b5-b6-track-video-models-20260923).

Lower-case **b1** and **b2** are the event-model networks trained for B1 and B2 (`b1_best.pt`, `b2_pretrained_best.pt`).
The B3–B5 bundles include both, and the P-stage reuses b1's fork head.

| Folder | Models | Main modules |
|---|---|---|
| `b12/` | B1, B2 event models | `prepare_events.py` (event samples from GT + detector), `mine_events.py` (hard negatives from real detector output), `cell_event.py` (3-frame residual 3D encoder, fork and edge heads, geometry features), `pretrained_event.py` (5-frame pretrained U-Net features, B2), `train_events.py`, `refine_events.py` (`EventRefiner`: crop, embed and score candidates at inference), `evaluate_events.py`, `align_training.py`, `prepare_clean_labels.py`, `split.json` |
| `b34/` | B3, B4 | `fast_division_review.py`; motion association (`motion_features.py`, `fast_motion_features.py`, `motion_refine.py`); image-embedding association for B4 (`visual_*`); 7-frame missed-cell recovery with an independent verifier (`recover_joint.py`, `verified_recovery.py`, `multiframe_recovery.py`); centre-offset refinement (`centroid_*`); the stage runners (`lineage_pipeline.py`, `parallel_pipeline.py`, `streaming_pipeline.py`); inference-time performance patches (`runtime_*_patch.py`) |
| `b56/` | B5, B6 | `track_video.py` / `train_track_video.py` (3D crop + 10-step trajectory transformer encoder); `prepare_corrupted_tracks.py` (wrong-link, broken-track and false-division augmentation); `track_set.py` / `train_track_set.py` (candidate-set attention); `train_track_rank_robust_matched.py` (3-seed LambdaRank); `track_*_refine.py` (graph-edit interfaces); `evaluate_track_stage.py`; `freeze_*` (frozen selection); `final_selection.json`, `*_training_protocol.json`, `release_acceptance_b56.json`, `submission_decision_b56.json` |

`b56/cloud_baseline.py` is the public baseline notebook's pipeline code, used to regenerate training graphs on cloud
machines. It is third-party, Apache-2.0 code with my modifications; see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

**Status of these models.** B3 → B5 improved calibration by about 0.001 and the audit by about 0.001. B3–B6 all scored 0.965
public and 0.928 private. `release_acceptance_b56.json` records that B5/B6 did **not** pass my original edge-quality release
gate. They were submitted on a composite-score decision, recorded in `submission_decision_b56.json`.
