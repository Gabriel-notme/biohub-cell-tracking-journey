# P-stage: post-lineage correction stage

This folder mirrors the Kaggle dataset `biohub-p12-division-stage` that every P-series notebook mounts, minus the DivNet
weights. Apart from this README, the files are byte-identical to the deployed ones. I checked all 69 against the
`stage_files_sha256` prefixes recorded by the final Kaggle run. The full specification with evidence is in [`docs/PIPELINE.md`](../docs/PIPELINE.md).

## Entry points

| Script | Used by | Adds |
|---|---|---|
| `p_stage.py` | P1, P2 | division completion + isolated-node pruning |
| `p_stage2.py` | P3, P4 | + dropped-sister recovery, double-fork resolution |
| `p_stage3.py` | P5, P6 | + learned relinking and free-end linking |
| `p_stage4.py` | P7, P8, P11, P13 | + isolated-node pruning after linking |
| `p_stage6.py` | P9, P10 | + junk-track pruning (not submitted) |
| `p_stage8.py` | P14 | + term_trim, long_link |
| `p_stage9.py` | P15 | + tbext, relinefit |
| `p_stage10.py`, `p_stage11.py` | P16, P17, P18 | + DivNet hook, P17 clean-up |
| **`p_stage12.py`** | P19 family, **P20, P21**, P23C, P23D | + robust clean-up (P19 family) |
| `p_stage13.py` | P22A, P22B (and the P23A/P23B previews) | + global-jump compensated relinking |

The command line:

```text
python p_stage12.py --artifact <B5 artefact bundle> --config <config.json> --data <dir of *.zarr>
                    --graphs <lineage_graphs> [--full <fullgraphs>] --out <out graphs> --work <work dir>
                    --submission <submission.csv> [--deadline <unix time>]
```

- The reference graphs are read from `<graphs>/../reference_graphs/`; there is no flag for them.
- `--full` is optional. Leaving it out skips every step that needs the candidate graph.
- `p_stage.py` (P1/P2) has no `--full` at all.
- A stage script silently ignores config keys it does not implement, so run each config with the script listed above.

## Modules

| Module | Step |
|---|---|
| `div_complete.py` | division completion with the b1 fork head (start / stolen candidates) |
| `dsr.py` | dropped-sister recovery from pre-ILP detections |
| `dfork.py` | double-fork resolution within K frames |
| `prune.py` | isolated-node pruning |
| `relink.py` + `relink_lgb.json` | learned relinking |
| `edge_link.py` + `edge_lgb.json` | learned free-end linking (gap 1 / gap 2) |
| `lgb_np.py` | dependency-free predictor for LightGBM JSON dumps |
| `frag_reinsert.py` | temporary fragment re-insertion (experimental, unused in submissions) |
| `jprune.py` + `jp_lgb.json` | junk-track pruning (P9 / P10 only; failed cross-embryo) |
| `p14_post.py` | term_trim, long_link |
| `p15_post.py` | tbext, relinefit |
| `p17_post.py` | cutdup, forkfrag, shortbranch |
| `p19_dup.py`, `p19_edge_deploy.py` | duplicate heads, term_trim re-run, parallel duplicates, FOV-border stubs |
| `p22_jump.py` | global-jump compensated relinking |
| `divnet_core.py`, `divnet16.py` | DivNet second opinion (P16 / P18). The weights are in the Kaggle dataset, not in this folder; without them the DivNet configs abort the whole P-stage run at worker start. The usage example in the `divnet_core.py` docstring is out of date: the installer is `divnet16.install(...)`, deployed with K=100. |

The `p_stage*.py` workers import `refine_events.EventRefiner`, and `div_complete.py` / `dsr.py` import `cell_event`. Both
come from the B5 bundle passed as `--artifact`; their source is in [`lineage_models/b56/`](../lineage_models/b56).

The `/workspace/...` paths near the top of `p19_dup.py` (the `REF` table and a `sys.path` entry) are research leftovers. The
deployed `post()` function does not depend on them.

## Configs

| Config | Variant |
|---|---|
| `p21s85_config.json` | **P21** (final selection) |
| `p20_config.json` | **P20** (final selection) |
| `p7_config.json` | **P7** (best private score, 0.939) |

Every other variant has its own `pN_config.json`, except P12, whose config and stage script were drafts and are not
included. `docs/PIPELINE.md` §5 lists them all. `vs90`, `vs95` and `vst80` are
threshold sweeps from the P3 period.
