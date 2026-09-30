"""(1) which fork par_dup breaks first under compression and how (daughter head / parent tail);
(2) per-movie density (same-frame NN distance, rounded coords) to see how much local movies vary;
(3) fork clearance vs par_dup: for every P15 fork, smallest radius at which par_dup (run >= 3) would pair a fork daughter (head) or the
    fork parent (tail) with a non-sister track; (4) gap-edge fork daughters: sister distance (cutdup risk)."""
import sys
sys.path.insert(0, '/workspace/cl/p16/redteam_p20')
from rt_common import *
from multiprocessing import Pool
from scipy.spatial import cKDTree
S0 = np.array([1.625, 0.40625, 0.40625])


def one(sm):
    s, m = sm
    n15, e15, _ = load(p15p(s, m))
    out = defaultdict(list); par = {}
    for a, b in pairs(e15): out[a].append(b); par[b] = a
    rpos = {n: np.array([max(0, int(round(float(v[k])))) for k in 'zyx']) * S0 for n, v in n15.items()}
    fpos = {n: np.array([float(v[k]) for k in 'zyx']) * S0 for n, v in n15.items()}
    byt = defaultdict(list)
    for n, v in n15.items(): byt[int(v['t'])].append(n)
    trees = {t: (ns, cKDTree(np.stack([rpos[n] for n in ns]))) for t, ns in byt.items()}
    nn = []
    for t, (ns, tr) in trees.items():
        if len(ns) > 1:
            d, _ = tr.query(np.stack([rpos[n] for n in ns]), k=2); nn.extend(d[:, 1].tolist())
    nn = np.array(nn)
    r = dict(set=s, movie=m, n=len(n15), nn_med=float(np.median(nn)), nn_p5=float(np.percentile(nn, 5)), nn_p1=float(np.percentile(nn, 1)),
             close35=float((nn < 3.5).mean()), nodes_per_frame=len(n15) / len(byt))
    # fork clearance: for each fork, daughters d1,d2: for first 3 frames along each daughter's linear path, the distance to the nearest node that is
    # not on the sister's path; run clearance = max over the 3 frames (par_dup needs all 3 frames within r). Same for the parent's last 3 frames.
    def path_fwd(n, k):
        L = [n]
        while len(L) < k and len(out.get(L[-1], [])) == 1: L.append(out[L[-1]][0])
        return L
    def path_back(n, k):
        L = [n]
        while len(L) < k and L[-1] in par and len(out[par[L[-1]]]) == 1: L.append(par[L[-1]])
        return L
    clr = []
    for p, cs in out.items():
        if len(cs) != 2: continue
        sis = [set(path_fwd(c, 3)) for c in cs]
        best = []
        for i, c in enumerate(cs):
            P = path_fwd(c, 3)
            if len(P) < 3: continue
            excl = sis[1 - i] | set(P)
            ds = []
            for x in P:
                ns, tr = trees[int(n15[x]['t'])]
                cand = [ns[j] for j in tr.query_ball_point(rpos[x], 8.0) if ns[j] not in excl]
                ds.append(min([float(np.linalg.norm(rpos[x] - rpos[y])) for y in cand], default=99.))
            best.append(('head', max(ds)))
        P = path_back(p, 3)
        if len(P) >= 3:
            excl = set(P); ds = []
            for x in P:
                ns, tr = trees[int(n15[x]['t'])]
                cand = [ns[j] for j in tr.query_ball_point(rpos[x], 8.0) if ns[j] not in excl]
                ds.append(min([float(np.linalg.norm(rpos[x] - rpos[y])) for y in cand], default=99.))
            best.append(('tail', max(ds)))
        if best: clr.append(min(best, key=lambda z: z[1]))
    r['fork_clear'] = clr
    fl = {}
    for ed in e15: fl[(int(ed['source_id']), int(ed['target_id']))] = 'gc' if 'gap_closed' in ed else ('g2' if 'gap2_recovered' in ed else 'o')
    gd = []
    for p, cs in out.items():
        if len(cs) != 2: continue
        for i, c in enumerate(cs):
            if fl[(p, c)] != 'o':
                syn = 'gap_synthetic' in n15[c] or fl[(p, c)] == 'g2'
                oc = out.get(c, [])
                nxt = len(oc) == 1 and fl.get((c, oc[0])) in ('gc', 'g2')
                ns, tr = trees[int(n15[c]['t'])]
                dmin = min([float(np.linalg.norm(fpos[c] - fpos[ns[j]])) for j in tr.query_ball_point(fpos[c], 10.0) if ns[j] != c], default=99.)
                gd.append(dict(cutdup_eligible=bool(syn and nxt), dmin=dmin, sister=float(np.linalg.norm(fpos[c] - fpos[cs[1 - i]]))))
    r['gap_daughters'] = gd
    return r


if __name__ == '__main__':
    ms = movies()
    with Pool(6) as pool: rows = pool.map(one, ms, chunksize=1)
    Path('/workspace/cl/p16/redteam_p20/rt_c.json').write_text(json.dumps(rows))
    med = np.array([r['nn_med'] for r in rows]); p5 = np.array([r['nn_p5'] for r in rows]); c35 = np.array([r['close35'] for r in rows])
    npf = np.array([r['nodes_per_frame'] for r in rows])
    print('per-movie median NN um: min %.2f p5 %.2f median %.2f max %.2f' % (med.min(), np.percentile(med, 5), np.median(med), med.max()))
    print('per-movie p5 NN um: min %.2f median %.2f' % (p5.min(), np.median(p5)))
    print('per-movie frac nodes with NN < 3.5um: max %.3f%% median %.3f%%' % (100 * c35.max(), 100 * np.median(c35)))
    print('nodes/frame: min %.0f median %.0f max %.0f' % (npf.min(), np.median(npf), npf.max()))
    for pre in ('44b6', '6bba'):
        mm = [r for r in rows if r['movie'].startswith(pre)]
        print(' %s: n=%d median NN med %.2f (min %.2f) nodes/frame median %.0f max %.0f' % (pre, len(mm), np.median([r['nn_med'] for r in mm]), min(r['nn_med'] for r in mm), np.median([r['nodes_per_frame'] for r in mm]), max(r['nodes_per_frame'] for r in mm)))
    cl = [c for r in rows for c in r['fork_clear']]
    v = np.array([c[1] for c in cl])
    print('forks with >=3-frame path: %d; run-clearance (min over head/tail of max-over-3-frames NN to non-sister): min %.2f p1 %.2f p5 %.2f median %.2f' % (len(v), v.min(), np.percentile(v, 1), np.percentile(v, 5), np.median(v)))
    for th in (3.5, 3.75, 4.0, 4.5, 5.0):
        print('  forks with run-clearance <= %.2f um: %d (head %d, tail %d)' % (th, int((v <= th).sum()), sum(1 for c in cl if c[1] <= th and c[0] == 'head'), sum(1 for c in cl if c[1] <= th and c[0] == 'tail')))
    gd = [g for r in rows for g in r['gap_daughters']]
    el = [g for g in gd if g['cutdup_eligible']]
    print('gap-edge fork daughters %d, cutdup-eligible (synthetic, gap in+out) %d; their NN dist (unrounded) sorted: %s' % (len(gd), len(el), sorted(round(g['dmin'], 2) for g in el)))
