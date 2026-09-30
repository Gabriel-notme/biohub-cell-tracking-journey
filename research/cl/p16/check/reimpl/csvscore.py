"""Rescore a submission CSV with review.score, but writing rows into this check dir and with at most 6 workers."""
import sys, os
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl')
from pathlib import Path
import multiprocessing
import review
review.REV = Path('/workspace/cl/p16/check/reimpl/rev'); review.REV.mkdir(exist_ok=True)
review.Pool = lambda n: multiprocessing.Pool(min(n, 6))
if __name__ == '__main__':
    review.score(sys.argv[1], sys.argv[2], sys.argv[3])
