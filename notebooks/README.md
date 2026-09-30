# Submitted Kaggle notebooks

These are the notebooks I submitted, with no outputs. The code cells are exactly as pushed. In the P21 description cell I
shortened one internal phrase; nothing else was edited.

| Folder | Variant | Public | Private | Notes |
|---|---|---:|---:|---|
| `p21-final-selection/` | P21 | 0.976 | 0.937 | selected for the final score |
| `p20-final-selection/` | P20 | 0.975 | 0.936 | selected for the final score |
| `p07-private-best/` | P7 | 0.972 | **0.939** | my best private score; not selected |

All three share cells 1–3: the frozen B5 lineage pipeline on the public baseline. Apart from the description cell, they
differ only in cell 4, which sets the P-stage script, the config file, the variant label and the status fields recorded in
`pstage_status.json`. P20 and P21 differ only in the description cell, the config file and the label.

Cell 2 is derived from the public Apache-2.0 notebook by zhincez and the notebooks it was built on; see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for the attribution chain and the statement of changes.

`build_variant.py` creates a new variant from any of them. It changes only the config name, the variant label and the
description, and refuses to change anything else. It also refuses a config that uses keys the notebook's stage script does
not implement. For example, P7's `p_stage4.py` has no P14+ steps, and P20/P21's `p_stage12.py` has no `p22jump`. Building
P23C from P21 reproduces the code cells of the submitted P23C notebook exactly.

`kernel-metadata.json` lists the required inputs:

- the competition;
- three public datasets by pilkwang;
- my B5/B6 bundle;
- my P-stage dataset (this repository's `pstage/`).

They run with GPU T4 ×2, internet off, and the Docker image pinned by digest. See
[`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).
