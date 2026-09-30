"""Preprocess: per movie (a) 2x2 xy-avg-pooled, quantile-normalised uint8 volume (T,Z,128,128) -> /dev/shm/linknet/vol/<movie>.npy
(b) P15 node table + rl_cands evaluable pair table -> /workspace/cl/p16/linknet/tab/<set>__<movie>.pkl"""
import os, sys, glob, json, pickle, time
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
import numpy as np
from multiprocessing import Pool
VOL = '/dev/shm/linknet/vol'; TAB = '/workspace/cl/p16/linknet/tab'


def job(f):
    import numcodecs.blosc
    numcodecs.blosc.use_threads = False
    import zarr
    s_, name = os.path.basename(f)[:-4].split('__')
    t0 = time.time()
    ov = '%s/%s.npy' % (VOL, name)
    if not os.path.exists(ov):
        g = zarr.open_group('/workspace/data/train/%s.zarr' % name, mode='r'); a = g['0']
        qs = g.attrs.get('image_statistics', {}).get('quantiles', {})
        lo = float(qs.get('0.001', 0)); hi = float(qs.get('0.999', 0))
        T, Z, Y, X = a.shape
        out = np.empty((T, Z, Y // 2, X // 2), np.uint8)
        for t in range(T):
            v = np.asarray(a[t], np.float32)
            if t == 0 and hi <= lo: lo, hi = np.percentile(v[::2, ::4, ::4], [.1, 99.9])
            v = v.reshape(Z, Y // 2, 2, X // 2, 2).mean((2, 4))
            v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 3) * 85.
            out[t] = np.rint(v).astype(np.uint8)
        np.save(ov + '.tmp.npy', out); os.rename(ov + '.tmp.npy', ov)
    ot = '%s/%s__%s.pkl' % (TAB, s_, name)
    if not os.path.exists(ot):
        gj = json.load(open('/workspace/cl/ps_p15_%s/graphs/%s.json' % (s_, name)))
        ids = np.array(sorted(int(k) for k in gj['nodes'])); idx = {n: i for i, n in enumerate(ids)}
        zyx = np.array([[gj['nodes'][str(n)]['z'], gj['nodes'][str(n)]['y'], gj['nodes'][str(n)]['x']] for n in ids], np.float32)
        tt = np.array([gj['nodes'][str(n)]['t'] for n in ids], np.int32)
        par = -np.ones(len(ids), np.int64); nch = np.zeros(len(ids), np.int32); ch1 = -np.ones(len(ids), np.int64)
        for e in gj['edges']:
            a_, b_ = idx[int(e['source_id'])], idx[int(e['target_id'])]; par[b_] = a_; nch[a_] += 1; ch1[a_] = b_
        ch1[nch != 1] = -1
        z = pickle.load(open(f, 'rb')); L = z['L']
        pairs = {}
        def add(x, y, tp, v):
            if x < 0 or y < 0 or not v: return
            k = (idx[int(x)], idx[int(y)])
            if k not in pairs: pairs[k] = int(tp)
        for i in range(len(L['s'])):
            add(L['s'][i], L['d'][i], L['tp_sd'][i], L['v_sd'][i]); add(L['s'][i], L['c'][i], L['tp_sc'][i], L['v_sc'][i])
            add(L['q'][i], L['d'][i], L['tp_qd'][i], L['v_qd'][i]); add(L['q'][i], L['c'][i], L['tp_qc'][i], L['v_qc'][i])
        # all pairs needed for scoring rows (incl. non-evaluable ones)
        need = set()
        m = lambda x: idx[int(x)] if x >= 0 else -1
        rows = np.array([[m(L['s'][i]), m(L['d'][i]), m(L['c'][i]), m(L['q'][i])] for i in range(len(L['s']))], np.int64).reshape(-1, 4)
        for s, d, c, q in rows:
            need.add((s, d))
            if c >= 0: need.add((s, c))
            if q >= 0: need.add((q, d))
            if c >= 0 and q >= 0: need.add((q, c))
        need = np.array(sorted(need), np.int64).reshape(-1, 2)
        P = np.array(sorted(pairs), np.int64).reshape(-1, 2); Y = np.array([pairs[tuple(k)] for k in P], np.int8)
        y = ((L['tp_sd'] == 1) & (L['tp_sc'] == 0) & (L['tp_qd'] == 0)).astype(np.int8)
        pickle.dump(dict(ids=ids, zyx=zyx, t=tt, par=par, ch1=ch1, nch=nch, P=P, Y=Y, need=need, rows=rows, y=y,
                         dtp=(L['tp_sd'] - L['tp_sc'] - L['tp_qd']).astype(np.int8),
                         dfp=((L['v_sd'] - L['tp_sd']) - (L['v_sc'] - L['tp_sc']) - (L['v_qd'] - L['tp_qd'])).astype(np.int8),
                         typ=z['X'][:, 0].astype(np.int8)), open(ot, 'wb'))
    return name, round(time.time() - t0, 1)


if __name__ == '__main__':
    fs = sorted(glob.glob('/workspace/cl/nm/rl_cands/*.pkl'))
    if len(sys.argv) > 1: fs = fs[:int(sys.argv[1])]
    with Pool(int(os.environ.get('NPROC', '40')), maxtasksperchild=4) as p:
        for r in p.imap_unordered(job, fs): print(*r, flush=True)
