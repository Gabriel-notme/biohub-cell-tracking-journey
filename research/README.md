# Research scripts

This folder is the working code of the P-series exploration, published as it was run, without clean-up or refactoring:
about 650 scripts from 24–29 September 2026. The scripts assume the directory layout of my cloud machines (`/workspace/...`)
and the official metric package on the path. They are here so that every number in `docs/` can be traced back to its code,
not as a library.

The `NOTES.md` files and `*.txt` result dumps are unedited working logs written in internal shorthand.

| Folder | What is in it |
|---|---|
| `code/` | P1–P4 period: division-completion prototypes (`div_complete*.py`, `dsr_*.py`, `dfork.py`), error decomposition (`gt_*`, `miss_*`, `trace_miss*`), early DivNet (`divnet_*`), notebook builders (`build_kernel*.py`, `run_nb*.py`, `lin_run.py`) |
| `cl/` (top level) | P5–P21 work, grouped below this table |
| `cl/ideas/` | rule modules evaluated with `rule_eval.py`: rule families from brainstorming rounds 2 and 3 (`r2_*`, `r3_*`; `r3_tbext.py` and `r3_pl_relinefit.py` became P15), P19 clean-up (`p19*`), P21/P22 ideas (`p21.py`, `p22_*.py`: frozen frames, jumps, jump oracle, flow and probability relinking) |
| `cl/nm/` | "new methods" round, each with an oracle bound and LOEO: learned verifier (`ver_*`), graph re-scoring (`gr_*`), coordinate refinement (`loc_*`), division learning (`dv_*`), relinking (`rl_*`) |
| `cl/p16/` | upstream model attempts: DivNet (`divnet/`, `divnet2/`), appearance link CNN (`linknet/`), b1 ensembles (`b1ens/`), independent checks of P17 / P19 (`check/`, `check2/`), deployment builders (`deploy/`). The model folders (`divnet/`, `divnet2/`, `linknet/`, `b1ens/`) each have a `NOTES.md` |
| `cl/p21/` | division-threshold bet: pre-registration (`PREREG_P21.txt`), fork-by-fork diff (`div_diff.py`), timing and z-bias studies |
| `nbsrc/` | harness that reruns the Kaggle notebook's base pipeline off-Kaggle: `gen_run.py`, `lin_run.py`, `run_ps_nb.sh`, B5-level knob experiments (`chain*.sh`, `early.py`), frozen-frame analysis (`frames_an*.py`) and the full comparison tool `cmp_all.py`. Some helpers they call are not included; see `docs/REPRODUCIBILITY.md` §3 |

The top level of `cl/` groups roughly as follows:

- **evaluation:** `review.py`, `rule_eval.py`, `strict.py`, `loeo.py`, `verify_kout.py`, `lb_noise.py`;
- **linker training:** `edge_cands.py`, `relink_train.py`, `edge_train.py`, `export_models.py`, `lgb_equiv.py`;
- **junk-track pruning:** `jp_*.py`;
- **division studies:** `div_*.py`, `fork_*.py`, `b1_calib.py`, `vparent*.py`, `ft_*.py`, `dc_bins.py`, `dc_type.py`;
- **oracle bounds:** `link_oracle*.py`, `gap_pot*.py`;
- **rejected rule families:** `shadow.py`, `smooth.py`, `xrl*.py`, `swap*.py`, `nswap*.py`, `trk_*.py`, `zcov.py`.

**Not here:** data and per-movie outputs; model weights, except the linker JSONs in `pstage/`; and scripts that handled cloud
machines or credentials.
