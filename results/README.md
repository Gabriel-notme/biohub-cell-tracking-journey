# Results

- `submissions.csv` lists every submission with its public and private leaderboard score, the Kaggle notebook title and
  version, and a one-line description of what changed relative to its parent. Scores are as shown on Kaggle, rounded to 3
  decimals; private scores are from the private leaderboard after the deadline. My earliest five exploratory notebooks
  (public 0.836–0.927) are not listed, and I did not record private scores for v13 and v14.
- `splits/` holds the movie lists of the local validation groups (movie ids only):

  | File | Movies |
  |---|---|
  | `hold36.txt` | 36 |
  | `preview4.txt` (prev4) | 4 |
  | `audit32.txt` | 32 |
  | `t127a.txt` | 64 |
  | `t127b.txt` | 63 |
  | `train127.txt` | t127a ∪ t127b |

  See [`docs/VALIDATION.md`](../docs/VALIDATION.md).
