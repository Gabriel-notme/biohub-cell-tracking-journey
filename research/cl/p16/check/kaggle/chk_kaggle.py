import os, sys, json
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS','RAYON_NUM_THREADS','BLOSC_NTHREADS']: os.environ[k]='1'
sys.path.insert(0, '/workspace/cl')
from pathlib import Path
import review
OUT = Path('/workspace/cl/p16/check/kaggle/rev'); review.REV = OUT
import pandas as pd, numpy as np
CSV = {'kg15': '/workspace/kout/p15/submission.csv', 'kg17': '/workspace/kout/p17/submission.csv',
       'lc15': '/workspace/cl/ps_p15_prev4/submission.csv', 'lc17': '/workspace/cl/p16/ps_p17_prev4/submission.csv',
       'kgbase': '/workspace/kout/p17/submission_base.csv'}
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'score':
        for tag in sys.argv[2:]:
            review.score(tag, 'prev4', CSV[tag])
    elif mode == 'diff':
        def load(tag):
            df = pd.read_csv(CSV[tag]); out = {}
            for ds, g in df.groupby('dataset'):
                n = g[g.row_type == 'node']; e = g[g.row_type == 'edge']
                pos = {int(r.node_id): (int(r.t), round(float(r.z), 2), round(float(r.y), 2), round(float(r.x), 2)) for r in n.itertuples()}
                E = [(pos[int(a)], pos[int(b)]) for a, b in zip(e.source_id, e.target_id)]
                ids_e = [(int(a), int(b)) for a, b in zip(e.source_id, e.target_id)]
                out[ds] = dict(nodes=pos, edges=E, ide=ids_e)
            return out
        D = {t: load(t) for t in ['kg15', 'kg17', 'lc15', 'lc17']}
        from collections import Counter
        def stats(d):
            outdeg = Counter(a for a, b in d['ide'])
            return dict(n=len(d['nodes']), e=len(d['edges']), forks=sum(1 for v in outdeg.values() if v >= 2),
                        dupnode=len(d['nodes']) - len(set(d['nodes'].values())), dupedge=len(d['edges']) - len(set(d['edges'])))
        res = {}
        for ds in sorted(D['kg17']):
            r = {t: stats(D[t][ds]) for t in D}
            N = {t: set(D[t][ds]['nodes'].values()) for t in D}; E = {t: set(D[t][ds]['edges']) for t in D}
            r['kg17_vs_lc17'] = dict(nodes_only_kg=len(N['kg17'] - N['lc17']), nodes_only_lc=len(N['lc17'] - N['kg17']),
                                     edges_only_kg=len(E['kg17'] - E['lc17']), edges_only_lc=len(E['lc17'] - E['kg17']),
                                     same_ids=D['kg17'][ds]['ide'] == D['lc17'][ds]['ide'])
            r['kg15_vs_lc15'] = dict(nodes_only_kg=len(N['kg15'] - N['lc15']), nodes_only_lc=len(N['lc15'] - N['kg15']),
                                     edges_only_kg=len(E['kg15'] - E['lc15']), edges_only_lc=len(E['lc15'] - E['kg15']))
            # P17 delta on kaggle vs local
            kd_rm_n = N['kg15'] - N['kg17']; kd_add_n = N['kg17'] - N['kg15']; ld_rm_n = N['lc15'] - N['lc17']; ld_add_n = N['lc17'] - N['lc15']
            kd_rm_e = E['kg15'] - E['kg17']; kd_add_e = E['kg17'] - E['kg15']; ld_rm_e = E['lc15'] - E['lc17']; ld_add_e = E['lc17'] - E['lc15']
            r['delta'] = dict(kg_rm_nodes=len(kd_rm_n), kg_add_nodes=len(kd_add_n), lc_rm_nodes=len(ld_rm_n), lc_add_nodes=len(ld_add_n),
                              rm_nodes_common=len(kd_rm_n & ld_rm_n), kg_rm_e=len(kd_rm_e), kg_add_e=len(kd_add_e), lc_rm_e=len(ld_rm_e), lc_add_e=len(ld_add_e),
                              rm_e_common=len(kd_rm_e & ld_rm_e), add_e_common=len(kd_add_e & ld_add_e))
            res[ds] = r
            print(ds, json.dumps(r))
        json.dump(res, open(OUT / 'diff.json', 'w'), indent=1)
