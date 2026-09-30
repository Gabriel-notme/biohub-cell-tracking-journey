"""How is the sparse GT distributed? Per movie: GT bbox (t,z,y,x), #nodes, #roots, root times, vs predicted (B5) node extent.
Also: fraction of predicted nodes inside the GT bbox (dilated), i.e. how many nodes are 'free' to remove in the oracle sense."""
import os, sys, json, glob
import numpy as np
import zarr
from pathlib import Path
from multiprocessing import Pool
SCALE = np.array([1.625, 0.40625, 0.40625])
SETS = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
        'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
        't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def read_gt(name):
    g = zarr.open_group('/workspace/data/train/%s.geff' % name, mode='r')
    ids = np.asarray(g['nodes/ids'])
    P = {k: np.asarray(g['nodes/props/%s/values' % k]) for k in 'tzyx'}
    E = np.asarray(g['edges/ids'])
    meta = dict(g.attrs)
    ntot = (meta.get('geff', {}).get('extra') or {}).get('estimated_number_of_nodes')
    return ids, P, E, ntot


def job(args):
    s, f = args
    name = Path(f).stem
    try:
        ids, P, E, ntot = read_gt(name)
    except Exception as e:
        return {'movie': name, 'set': s, 'err': str(e)}
    d = json.load(open(f)); nodes = d['nodes']
    pt = np.array([[v['t'], v['z'], v['y'], v['x']] for v in nodes.values()], float)
    gt = np.stack([P['t'], P['z'], P['y'], P['x']], 1).astype(float)
    par = {int(b): int(a) for a, b in E}
    roots = [i for i in ids.tolist() if int(i) not in par]
    idx = {int(i): k for k, i in enumerate(ids.tolist())}
    rt = sorted(int(gt[idx[r], 0]) for r in roots)
    lo, hi = gt.min(0), gt.max(0)
    out = {'movie': name, 'set': s, 'ntot': ntot, 'n_gt': len(ids), 'n_gt_edges': len(E), 'n_pred': len(pt), 'n_roots': len(roots), 'root_t': rt[:40],
           'gt_lo': lo.tolist(), 'gt_hi': hi.tolist(), 'pred_lo': pt.min(0).tolist(), 'pred_hi': pt.max(0).tolist()}
    # fraction of predicted nodes inside GT bbox dilated by m micrometres (space) and within t range
    for m in [0, 10, 20, 40]:
        dz, dy = m / SCALE[0], m / SCALE[1]
        inside = (pt[:, 0] >= lo[0]) & (pt[:, 0] <= hi[0]) & (pt[:, 1] >= lo[1] - dz) & (pt[:, 1] <= hi[1] + dz) & (pt[:, 2] >= lo[2] - dy) & (pt[:, 2] <= hi[2] + dy) & (pt[:, 3] >= lo[3] - dy) & (pt[:, 3] <= hi[3] + dy)
        out['in_bbox_%d' % m] = float(inside.mean())
    # GT nodes per frame, pred nodes per frame
    out['gt_per_t'] = np.bincount(gt[:, 0].astype(int)).tolist()
    return out


if __name__ == '__main__':
    jobs = [(s, f) for s, d in SETS.items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(64) as p: R = p.map(job, jobs)
    json.dump(R, open('/workspace/cl/gt_region.json', 'w'))
    R = [r for r in R if 'err' not in r]
    print('movies', len(R))
    for k in ['n_gt', 'n_gt_edges', 'n_pred', 'n_roots', 'ntot']:
        v = np.array([r[k] for r in R], float); print(k, 'median %.0f min %.0f max %.0f' % (np.median(v), v.min(), v.max()))
    for m in [0, 10, 20, 40]:
        v = np.array([r['in_bbox_%d' % m] for r in R]); print('pred frac in GT bbox +%dum: median %.3f mean %.3f' % (m, np.median(v), v.mean()))
    rt0 = np.array([np.mean(np.array(r['root_t']) == 0) for r in R]); print('frac roots at t=0: median %.2f mean %.2f' % (np.median(rt0), rt0.mean()))
    ext = np.array([(np.array(r['gt_hi']) - np.array(r['gt_lo'])) * np.r_[1, SCALE] for r in R]); print('GT extent t,z,y,x (um) median', np.median(ext, 0).round(1), 'p90', np.percentile(ext, 90, 0).round(1))
    pext = np.array([(np.array(r['pred_hi']) - np.array(r['pred_lo'])) * np.r_[1, SCALE] for r in R]); print('pred extent median', np.median(pext, 0).round(1))
    for r in R[:6]: print(r['movie'], r['set'], r['n_gt'], r['n_roots'], r['root_t'][:10], [round(x) for x in r['gt_lo']], [round(x) for x in r['gt_hi']], [round(x) for x in r['pred_lo']], [round(x) for x in r['pred_hi']], r['gt_per_t'][:5], r['gt_per_t'][-3:])
