# Reproducing the submissions and the local evaluation

## 1. What is in this repository and what is not

**Included:**

- The P-stage as deployed (`pstage/`): every stage script and config in the deployed Kaggle dataset, and the LightGBM
  linker models exported to JSON.
- The three notebooks that matter (P7, P20, P21), as submitted.
- All B1–B6 lineage-model source code, the validation splits and the frozen training protocols (`lineage_models/`).
- Research and evaluation scripts, as run (`research/`).
- The movie splits used for validation (`results/splits/`).

**Not included:**

- Competition data. It is released under CC0; download it from Kaggle.
- Weights of the public detector, DeepCenter and support pack. They are public CC0 Kaggle datasets by pilkwang; see
  `THIRD_PARTY_NOTICES.md`.
- Weights of my B5/B6 lineage bundle (the b1 event model, track-video encoder, set-attention and LambdaRank heads), and the
  DivNet weights (P16 / P18 only). Both are in my public Kaggle datasets below.
- Scripts that provisioned cloud machines or handled credentials.

The notebooks load two of my Kaggle datasets, both public:

- [`shawsebastian/biohub-b5-b6-track-video-models-20260923`](https://www.kaggle.com/datasets/shawsebastian/biohub-b5-b6-track-video-models-20260923), the frozen B5/B6 bundle with all
  lineage-model weights, including b1 (Apache-2.0, because it contains the Apache-derived `cloud_baseline.py`);
- [`shawsebastian/biohub-p12-division-stage`](https://www.kaggle.com/datasets/shawsebastian/biohub-p12-division-stage), identical to `pstage/` plus the DivNet weights (CC0).

Attach them directly to a Kaggle notebook, or download them with `kaggle datasets download <owner>/<slug>`.

## 2. Running a submission notebook on Kaggle

1. **Attach my public P-stage dataset**, or create your own from `pstage/` with the slug `biohub-p12-division-stage`, for
   example with `kaggle datasets create -p pstage` after adding a `dataset-metadata.json` with your own id. Cell 4 looks for
   exactly this slug (`P_SLUG`). If it is missing, the notebook silently submits the B5 base.
2. **Attach the inputs** to a copy of `notebooks/p21-final-selection/model.ipynb`:
   - the competition;
   - the three pilkwang datasets listed in its `kernel-metadata.json`;
   - the B5 bundle;
   - your P-stage dataset.
3. **Check the owner paths.** Cell 1 looks for the bundle under `/kaggle/input/datasets/<owner>/<slug>` and
   `/kaggle/input/<slug>`, and cell 4 does the same for the P-stage dataset. With my public datasets attached, the paths work
   as they are. If you use your own copies, change the owner `shawsebastian` to yours. Keep the manifest hash assertions in
   cell 1 intact.
4. **Configure and run.** Use GPU T4 ×2 with internet off, then Save & Run. A preview run takes 27–30 minutes.
5. **Check `pstage_status.json`.**
   - It should show `applied: true` and the expected stage-file hash prefixes.
   - `errors` counts only movies that fell back completely. Also check that `dsr_errors` (which also counts double-fork
     errors), `link_errors`, `p14_errors` and `p15_errors` are 0.
   - P17/P19 failures appear only in the per-movie records of `/kaggle/working/pstage/pstage_report.json` (see §3).
6. **Submit that kernel version** to the competition. Scoring the full hidden set takes about 9 hours.

To build another variant (a different config), use `notebooks/build_variant.py`:

- It changes only the config name, the variant label and the description, and refuses anything else.
- It also refuses configs that use keys the notebook's stage script does not implement:
  - the P20 / P21 notebooks run `p_stage12.py`, which has no `p22jump` or `junk_prune`;
  - the P7 notebook runs `p_stage4.py`, which has nothing from P14 onwards.
- P22A/B and P23A/B need cell 4 edited to call `p_stage13.py`.
- `dataset_sources` in `kernel-metadata.json` still name my datasets; edit them by hand.

```bash
python notebooks/build_variant.py notebooks/p21-final-selection p23c_config.json P23C my-p23c "My P23C" "P21 with start threshold 0.80" <your-kaggle-user>
```

## 3. Running the P-stage locally

The P-stage needs these inputs for each movie:

| Input | Produced by | Location |
|---|---|---|
| B5 lineage graph | cell 3 | `/kaggle/working/lineage_graphs/<movie>.json` → `--graphs` |
| reference graph (before the lineage stages) | cell 2 | `/kaggle/working/reference_graphs/<movie>.json`; read from `<graphs>/../reference_graphs/`, with no flag |
| pre-ILP candidate graph | cell 2 | `/kaggle/working/fullgraphs/<movie>.geff` → `--full` |
| image | competition data | the competition's test (or train) directory → `--data` |
| B5 artefact bundle | B5 dataset | `/kaggle/input/<slug>[/artifact_bundle]`, or `/kaggle/working/frozen_lineage_artifact` when zipped → `--artifact` |

Keep `lineage_graphs/`, `reference_graphs/` and `fullgraphs/` side by side. Without `reference_graphs/`, relinefit and the
P17/P19 clean-up fail for every movie and are skipped. `--data` must contain exactly the movies that have graphs; use a
folder of symlinks per set.

To produce these inputs for labelled training movies, run the notebook's cells 1–3 on a GPU machine with `/kaggle/input`
symlinked to local copies. `research/nbsrc/` holds the harness I used:

- `gen_run.py` turns cells 1–3 into a script, remaps `/kaggle/working` and lets you override `BIOHUB_*` knobs. It expects
  the cells extracted to `cell1.py`–`cell3.py`, for example by writing `"".join(nb["cells"][i]["source"])` for i = 1..3 from
  `notebooks/p20-final-selection/model.ipynb`.
- `lin_run.py` runs the lineage stages in parallel.
- `run_ps_nb.sh` runs the P-stage and scores it. It calls `p_stage12_gpuoff.py`, a local copy of `pstage/p_stage12.py` with a
  CUDA device offset that is not included; use `pstage/p_stage12.py` instead.
- `chain*.sh` and `resume_lin_ps.sh` also call helpers (`gen_run2.py`, `run_ps_h0.sh`) that are not included.

On the 99 movies I checked, the harness reproduced the original B5 scores to 5 decimals. It takes about 80 minutes per 100
movies on 2× H100.

Then run:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1
python pstage/p_stage12.py \
    --artifact <artifact_bundle> --config pstage/p21s85_config.json \
    --data <dir with the same movies' *.zarr> --graphs <working>/lineage_graphs --full <working>/fullgraphs \
    --out <out>/graphs --work <out>/work --submission <out>/submission.csv
```

The script shards movies over up to 2 GPUs, checks the graph after every optional step, and falls back per step and per
movie.

`BIOHUB_BASE_REPO` only needs to point at the baseline inference repository (which cell 2 unpacks to
`/kaggle/working/tracking_repo`) for event checkpoints that use pretrained U-Net features; b1 does not.

`<out>/work/pstage_report.json` summarises the counters: forks added, relinks, links, trims, tbext additions and errors.
Per-step failures are recorded per movie, and this should print `[]`:

```bash
python -c "import json;r=json.load(open('<out>/work/pstage_report.json'));print([(x['movie'],k) for x in r['records'] for k in x['pstage'] if k.endswith('_error') or k.endswith('_skipped')])"
```

The thread variables matter. Without them, process pools combined with BLAS, polars and blosc threads can exhaust a
container's process limit and deadlock.

## 4. Scoring with the official metric

1. Clone the organisers' metric at the pinned commit
   `https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f5a52d11077f9dc2b074644618f26939e2`
   and put its `src/` on `PYTHONPATH` (the package is `tracking_cellmot`).
2. `research/cl/review.py scoreg <tag> <set> <graph dir> <movie list>` writes per-movie rows and prints the aggregate score.
3. Comparisons (bootstrap CI, drop-top-k, per-embryo and per-set deltas, division / edge / node bookkeeping) come from three
   tools:
   - `research/nbsrc/cmp_all.py <candidate> <reference> ...`;
   - `research/cl/rule_eval.py`, which applies a rule module to saved graphs;
   - `research/cl/strict.py <rows.json> <vi> [label]`, the S1–S7 gate.
4. The research scripts assume the directory layout of my cloud machines (`/workspace/...`). Adjust the paths at the top of
   each script.

## 5. Training the P-stage linkers

- **Candidate generation and labelling:** `research/cl/edge_cands.py`, `research/cl/relink_train.py` and
  `research/cl/edge_train.py`. Labels come from official edge matching on t127a, t127b and audit32 only.
- **Export to JSON** for the dependency-free predictor: `research/cl/export_models.py`. Check equivalence with
  `research/cl/lgb_equiv.py`.
- **Leave-one-embryo-out evaluation:** `research/cl/loeo.py`.

## 6. Environment

- **Kaggle.** The notebook installs pinned wheels offline from the pilkwang support pack. The Docker image digest is in each
  `kernel-metadata.json`.
- **Local and cloud reference.** Python 3.12 and PyTorch 2.8 (CUDA 12.8), with NumPy, SciPy, zarr, polars and ilpy.
  LightGBM is needed for training only.
- **Deterministic reruns.** The P-stage is deterministic. Across T4 and H100, the upstream detector can differ by 1–4 nodes
  per movie; on the preview movies, edges and divisions matched exactly in every check I ran.
- **Line endings.** Files are committed byte for byte (`.gitattributes` disables end-of-line conversion). Some original
  scripts use CRLF, and several of them are hash-checked by the B5 bundle manifest.
