"""Critic-B probe (stage-ordering interaction): B5 linefit smoothing (weight 0.8, window 2) is computed in the output filter on the
REFERENCE-graph track structure, before B5's lineage stages and before every P-stage re-wire (div_complete, dfork, relink,
edge_link, term_trim join, long_link). Nodes whose final +-2 track neighbourhood differs were smoothed along stale neighbours.
Re-smooth them on the final structure: new = cur + w * 0.8 * (fit_final - fit_ref), inputs = pre-linefit (fullgraph raw) coords.
w=0 reproduces P14. Stats: sanity (linefit on ref structure reproduces ref coords), number of re-smoothed nodes, shift sizes."""
import sys, json
import builtins
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/workspace/p56stage')
WANTS_META = True
REF = {'hold36': '/workspace/runs/b5f_hold36/working/reference_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/reference_graphs',
       'audit32': '/workspace/sync3/runs/b5f_audit32/working/reference_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/reference_graphs',
       't127b': '/workspace/sync3/runs/b5f_t127b/working/reference_graphs'}
S = np.array([1.625, .40625, .40625]); SHAPE = np.array([64, 256, 256]); W = 0.8


def _struct(edges, t_of):
    pred = defaultdict(list); succ = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        if a in t_of and b in t_of and t_of[b] == t_of[a] + 1:
            succ[a].append(b); pred[b].append(a)
    return pred, succ


def _hood(n, pred, succ, have):
    h = [(0, n)]; c = n
    for k in range(1, 3):
        p = pred.get(c, [])
        if len(p) != 1: break
        c = p[0]
        if c not in have: break
        h.append((-k, c))
    c = n
    for k in range(1, 3):
        s = succ.get(c, [])
        if len(s) != 1: break
        c = s[0]
        if c not in have: break
        h.append((k, c))
    return tuple(sorted(h))


def _fit(h, orig):
    dts = np.array([d for d, _ in h], float); X = np.stack([orig[m] for _, m in h])
    md = dts.mean(); mx = X.mean(0); var = ((dts - md) ** 2).sum()
    slope = ((dts - md)[:, None] * (X - mx)).sum(0) / var
    return mx - slope * md


def _cat(h_ref, h_cur):
    if len(h_ref) < 3 and len(h_cur) >= 3: return 'grow'
    if len(h_ref) >= 3 and len(h_cur) < 3: return 'shrink'
    if len(h_ref) < 3 and len(h_cur) < 3: return 'none'
    a, b = builtins.set(h_ref), builtins.set(h_cur)
    if b > a: return 'extend'
    if a > b: return 'trunc'
    return 'stale'


B5L = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
       'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
       't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def apply(nodes, edges, name=None, set=None, fullgeff=None, zarr=None, w=1.0, minsep=2.0, cats=None, ins_cur=True, src='final', cen_aware=False,
          rand=False):
    """src: 'final' = re-smooth on the P14 structure; 'b5' = only B5 lineage-stage edits (B5 lineage graph structure);
    'pstage' = only P-stage edits (reference = B5 lineage structure, target = P14). cen_aware: w=1 for nodes the centroid stage left in place."""
    from scipy.spatial import cKDTree
    from edge_link import load_full
    R = json.load(open(REF[set] + '/' + name + '.json'))
    rn = {int(k): v for k, v in R['nodes'].items()}
    fids, fT, fV, fE, fprob = load_full(fullgeff)
    raw = {int(i): (int(t), v) for i, t, v in zip(fids.tolist(), fT.tolist(), fV)}
    # linefit input positions ("orig") for reference-graph nodes: raw detection coords when the id is a detection of the same frame
    orig = {}
    for k, v in rn.items():
        f = raw.get(k)
        rp = np.array([v[c] for c in 'zyx'], float)
        if f is not None and f[0] == int(v['t']) and np.linalg.norm((f[1] - rp) * S) <= 3.0: orig[k] = np.asarray(f[1], float)
        else: orig[k] = rp  # synthetic / unaligned: best available proxy
    t_ref = {k: int(v['t']) for k, v in rn.items()}
    pr, sr = _struct(R['edges'], t_ref)
    # sanity: does linefit on the reference structure reproduce the reference coords?
    ok = bad = 0
    for k in list(rn)[::50]:
        if k not in raw: continue
        h = _hood(k, pr, sr, orig)
        rp = np.array([rn[k][c] for c in 'zyx'], float)
        pos = orig[k] if len(h) < 3 else (1 - W) * orig[k] + W * _fit(h, orig)
        if np.linalg.norm((pos - rp) * S) < 1e-3: ok += 1
        else: bad += 1
    t_cur = {k: int(v['t']) for k, v in nodes.items()}
    pc, sc = _struct(edges, t_cur)
    if src in ('b5', 'pstage'):
        L = json.load(open(B5L[set] + '/' + name + '.json'))
        t_l = {int(k): int(v['t']) for k, v in L['nodes'].items()}
        pl, sl = _struct(L['edges'], t_l)
        if src == 'b5': pc, sc = pl, sl          # target = B5 lineage structure
        else: pr, sr = pl, sl                   # reference = B5 lineage structure, target = P14
    orig_c = dict(orig)
    if ins_cur:  # nodes added after the output filter (B5 recovered nodes, P-stage gap2 inserts) enter the fit at their current position
        for k, v in nodes.items():
            if k not in orig_c: orig_c[k] = np.array([v[c] for c in 'zyx'], float)
    have_c = builtins.set(k for k in nodes if k in orig_c)
    new = {k: dict(v) for k, v in nodes.items()}; changed = builtins.set(); shifts = []; cc = defaultdict(int)
    for k in nodes:
        if k not in orig: continue
        h_ref = _hood(k, pr, sr, orig); h_cur = _hood(k, pc, sc, have_c)
        if h_ref == h_cur: continue
        cat = _cat(h_ref, h_cur); cc[cat] += 1
        if cats is not None and cat not in cats: continue
        f_ref = orig[k] if len(h_ref) < 3 else (1 - W) * orig[k] + W * _fit(h_ref, orig)
        f_cur = orig[k] if len(h_cur) < 3 else (1 - W) * orig[k] + W * _fit(h_cur, orig_c)
        ww = w
        if cen_aware and k in rn and np.allclose([nodes[k][c] for c in 'zyx'], [rn[k][c] for c in 'zyx']): ww = 1.0
        dlt = ww * (f_cur - f_ref)
        if not np.any(dlt): continue
        if rand:  # control: same physical shift length, random direction (seeded per movie+node)
            import zlib
            rg = np.random.default_rng(zlib.crc32(('%s_%d_%s' % (name, int(k), rand)).encode()))
            u = rg.normal(size=3); u /= np.linalg.norm(u)
            dlt = u * np.linalg.norm(dlt * S) / S
        q = np.array([nodes[k][c] for c in 'zyx'], float) + dlt
        if np.any(q < 0) or np.any(np.rint(q) >= SHAPE): continue
        new[k].update(dict(zip('zyx', map(float, q)))); changed.add(k); shifts.append(float(np.linalg.norm(dlt * S)))
    frames = defaultdict(list)
    for k, v in nodes.items(): frames[int(v['t'])].append(k)
    rej_total = 0
    for ns in frames.values():
        if len(ns) < 2 or not any(n in changed for n in ns): continue
        old = np.array([[nodes[n][c] for c in 'zyx'] for n in ns]) * S
        for _ in range(3):
            pp = np.array([[new[n][c] for c in 'zyx'] for n in ns]) * S
            rej = builtins.set()
            for i, j in cKDTree(pp).query_pairs(minsep):
                if np.linalg.norm(pp[i] - pp[j]) + 1e-5 < min(minsep, np.linalg.norm(old[i] - old[j])):
                    rej.update(n for n in (ns[i], ns[j]) if n in changed)
            if not rej: break
            for n in rej: new[n] = dict(nodes[n]); changed.discard(n)
            rej_total += len(rej)
    sh = np.array(shifts) if shifts else np.zeros(1)
    st = {'rl_sanity_ok': ok, 'rl_sanity_bad': bad, 'rl_moved': len(changed), 'rl_rej': rej_total,
          'rl_sh_gt05': int((sh > 0.5).sum()), 'rl_sh_gt1': int((sh > 1.0).sum()), 'rl_sh_gt2': int((sh > 2.0).sum())}
    st.update({'rl_c_' + k: v for k, v in cc.items()})
    return new, edges, st
