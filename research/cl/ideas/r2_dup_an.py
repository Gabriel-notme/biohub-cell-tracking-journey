"""Read-only DUPLICATE lens on P14 (= P13 final graph + ideas.combo14 trim,long). For each candidate duplicate class, the removal is
applied to P14 alone and re-scored with the official metric; plus GT labels of the removed nodes/edges (matched, TP / evaluable FP /
non-evaluable) for mechanism. Classes (all GT-free):
  ee     END next to END (both stop) within r: drop the one with the shorter back-history (single node)
  ee_tt  ee, then p14 term_trim again (the pair's parents now form an END next to a continuing node)
  fd     fork-daughter END (branch length 1) within r of a same-frame node with a child (sibling or other)
  fdb    fork-daughter branch of length 2..5 ending, each node within r of a same-frame continuing non-branch node: drop branch
  i2     isolated 2-node component with a node within r of a same-frame node of another component: drop both
  zd     END (not fork daughter, backlen>=3) with nearest same-frame continuing node at 3.5<d<=5 um and dxy<=1 um: drop (iterated)
  bub    isolated linear component 3..10 nodes whose first AND last node are within r of other-component same-frame nodes: drop
  fsib   fork whose two daughters are within r of each other: cut the edge to the daughter with the smaller 10-frame subtree
  tt2    p14 term_trim re-applied after long_link
usage: python3 r2_dup_an.py"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
for p in ['/workspace/cl', '/workspace/cl/ideas', '/workspace/official/src', '/workspace/code', '/workspace/p56stage']:
    sys.path.insert(0, p)
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np

S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
R = 3.5
CLASSES = ['ee', 'ee_tt', 'fd', 'fdb', 'i2', 'zd', 'bub', 'fsib', 'tt2']


def struct(edges):
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    return out, par


def rpos(nodes):
    return {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}


def drop_nodes(nodes, edges, drop):
    return ({k: v for k, v in nodes.items() if k not in drop},
            [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop])


def comps(nodes, out, par):
    adj = defaultdict(list)
    for a, bs in out.items():
        for b in bs: adj[a].append(b); adj[b].append(a)
    cid = {}; members = []
    for n in nodes:
        if n in cid: continue
        st = [n]; cid[n] = len(members); cur = [n]
        while st:
            x = st.pop()
            for y in adj.get(x, []):
                if y not in cid: cid[y] = cid[n]; st.append(y); cur.append(y)
        members.append(cur)
    return cid, members


def gen(nodes, edges):
    from scipy.spatial import cKDTree
    out, par = struct(edges); pos = rpos(nodes)
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    trees = {t: (cKDTree(np.stack([pos[n] for n in ns])), ns) for t, ns in byt.items()}
    cid, members = comps(nodes, out, par)

    def near(n, r=R):
        tr, ns = trees[int(nodes[n]['t'])]
        return [ns[j] for j in tr.query_ball_point(pos[n], r) if ns[j] != n]

    def backlen(n):
        L = 1
        while n in par and len(out[par[n]]) == 1: n = par[n]; L += 1
        return L

    def is_end(n): return (n in par) and not out[n]
    def fork_d(n): return n in par and len(out[par[n]]) == 2
    D = {c: set() for c in CLASSES}; CUT = set(); meta = defaultdict(list)
    # ee
    pairs = []
    for n in nodes:
        if not is_end(n): continue
        for m in near(n):
            if m > n and is_end(m): pairs.append((float(np.linalg.norm(pos[n] - pos[m])), n, m))
    used = set()
    for d, a, b in sorted(pairs):
        if a in used or b in used: continue
        la, lb = backlen(a), backlen(b)
        x = a if (la, -a) < (lb, -b) else b
        used.update((a, b)); D['ee'].add(x)
        # parallel run: frames back where both chains stay within R
        p, q, run = a, b, 1
        while p in par and q in par:
            p, q = par[p], par[q]
            if np.linalg.norm(pos[p] - pos[q]) > R: break
            run += 1
        meta['ee'].append(dict(d=d, lshort=min(la, lb), llong=max(la, lb), run=run, fd=fork_d(x)))
    # fd / fdb
    for p_, cs in out.items():
        if len(cs) != 2: continue
        for c in cs:
            chain = [c]; x = c
            while len(out[x]) == 1 and len(chain) <= 6: x = out[x][0]; chain.append(x)
            if out[x] or len(chain) > 5: continue  # branch must END within 5 nodes
            sib = [y for y in cs if y != c][0]
            ok = True; sibnear = False
            for y in chain:
                nb = [m for m in near(y) if out[m] and m not in chain]
                if not nb: ok = False; break
                if y == c and sib in nb: sibnear = True
            if not ok: continue
            if len(chain) == 1: D['fd'].add(c); meta['fd'].append(dict(sib=sibnear))
            else: D['fdb'].update(chain); meta['fdb'].append(dict(L=len(chain), sib=sibnear))
    # i2
    for mem in members:
        if len(mem) != 2: continue
        a, b = mem
        oth = [m for y in mem for m in near(y) if cid[m] != cid[a]]
        if oth: D['i2'].update(mem); meta['i2'].append(dict(both=all(any(cid[m] != cid[a] for m in near(y)) for y in mem)))
    # bub
    for mem in members:
        if not (3 <= len(mem) <= 10): continue
        if any(len(out[y]) > 1 for y in mem): continue
        first = [y for y in mem if y not in par][0]; last = [y for y in mem if not out[y]][0]
        if any(cid[m] != cid[first] for m in near(first)) and any(cid[m] != cid[last] for m in near(last)):
            D['bub'].update(mem); meta['bub'].append(dict(L=len(mem)))
    # fsib
    def sub10(c):
        cnt = 0; fr = [(c, 0)]
        while fr:
            x, d = fr.pop(); cnt += 1
            if d < 10: fr += [(y, d + 1) for y in out[x]]
        return cnt
    for p_, cs in out.items():
        if len(cs) != 2: continue
        a, b = cs
        if np.linalg.norm(pos[a] - pos[b]) <= R:
            sa, sb = sub10(a), sub10(b)
            c = a if (sa, -a) < (sb, -b) else b
            CUT.add((p_, c)); meta['fsib'].append(dict(sa=min(sa, sb), sb=max(sa, sb)))
    return D, CUT, meta, out, par, pos, near, is_end, fork_d, backlen


def zd_trim(nodes, edges, iters=5):
    from scipy.spatial import cKDTree
    nodes = dict(nodes); edges = list(edges); tot = 0
    for it in range(iters):
        out, par = struct(edges); pos = rpos(nodes)
        byt = defaultdict(list)
        for n, v in nodes.items(): byt[int(v['t'])].append(n)
        drop = set()
        for t, ns in byt.items():
            if len(ns) < 2: continue
            tr = cKDTree(np.stack([pos[n] for n in ns]))
            for n in ns:
                if out[n] or n not in par or len(out[par[n]]) == 2: continue
                L, x = 1, n
                while x in par: x = par[x]; L += 1
                if L < 3: continue
                cand = [ns[j] for j in tr.query_ball_point(pos[n], 5.0) if ns[j] != n and out[ns[j]] and ns[j] not in drop]
                if not cand: continue
                if any(np.linalg.norm(pos[m] - pos[n]) <= R for m in cand): continue  # term_trim's job
                if any(np.linalg.norm((pos[m] - pos[n])[1:]) <= 1.0 for m in cand): drop.add(n)
        if not drop: break
        tot += len(drop); nodes, edges = drop_nodes(nodes, edges, drop)
    return nodes, edges, tot


def job(f):
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx, combo14, p14_post
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    name = Path(f).stem; s = f.split('ps_p13_')[1].split('/')[0]
    nodes, edges = evalx.load_graph_json(f)
    nodes, edges, _ = combo14.apply(nodes, edges, order='trim,long', fullgeff=FULL[s] + '/' + name + '.geff')
    base = evalx.score_movie(name, nodes, edges)
    # GT labels on P14
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs(); GE = set(zip([int(a) for a in ea[K.EDGE_SOURCE].to_list()], [int(b) for b in ea[K.EDGE_TARGET].to_list()]))
    gout = Counter(a for a, b in GE); gin = Counter(b for a, b in GE)

    def elab(a, b):
        ga, gb = p2g.get(a), p2g.get(b)
        if ga is not None and gb is not None and (ga, gb) in GE: return 'tp'
        if (ga is not None and gout[ga] > 0) or (gb is not None and gin[gb] > 0): return 'fpe'
        return 'ne'
    D, CUT, meta, out, par, pos, near, is_end, fork_d, backlen = gen(nodes, edges)
    res = {'movie': name, 'set': s, 'base': base, 'rows': {}, 'lab': {}, 'meta': dict(meta), 'n': {}}
    for c in CLASSES:
        if c == 'tt2':
            nn, ne, st = p14_post.term_trim(nodes, edges); cnt = st['trim']; drop = set(nodes) - set(nn)
        elif c == 'ee_tt':
            nn, ne = drop_nodes(nodes, edges, D['ee']); nn, ne, st = p14_post.term_trim(nn, ne); drop = set(nodes) - set(nn); cnt = len(drop)
        elif c == 'zd':
            nn, ne, cnt = zd_trim(nodes, edges); drop = set(nodes) - set(nn)
        elif c == 'fsib':
            ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in CUT]; nn = nodes; drop = set(); cnt = len(CUT)
        else:
            drop = D[c]; nn, ne = drop_nodes(nodes, edges, drop); cnt = len(drop)
        res['n'][c] = cnt
        if cnt == 0: res['rows'][c] = base; res['lab'][c] = {}; continue
        res['rows'][c] = evalx.score_movie(name, nn, ne)
        lab = Counter()
        for n in drop:
            lab['node_matched' if n in p2g else 'node_unmatched'] += 1
            if n in p2g: lab['node_gt_ann_out' if gout[p2g[n]] > 0 else 'node_gt_noout'] += 1
        rem = [(int(e['source_id']), int(e['target_id'])) for e in edges if int(e['source_id']) in drop or int(e['target_id']) in drop]
        if c == 'fsib': rem = list(CUT)
        for a, b in rem: lab['e_' + elab(a, b)] += 1
        res['lab'][c] = dict(lab)
    return res


if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(40, maxtasksperchild=4) as p: RR = p.map(job, fs, chunksize=1)
    json.dump(RR, open('/workspace/cl/ideas/r2_dup_an.json', 'w'))
    from tracking_cellmot.metrics import summarise
    rng = np.random.default_rng(0)
    ms = sorted(range(len(RR)), key=lambda i: RR[i]['movie'])
    print('movies', len(RR))
    K_ = [rng.integers(0, len(RR), len(RR)) for _ in range(300)]
    for c in CLASSES:
        line = '%-6s n=%5d' % (c, sum(r['n'][c] for r in RR))
        for emb in ['44b6', '6bba', '']:
            sel = [r for r in RR if r['movie'].startswith(emb)]
            a = summarise([r['base'] for r in sel]); b = summarise([r['rows'][c] for r in sel])
            line += ' | %s %+.5f (e %+.5f d %+d/%+d)' % (emb or 'all', b['score'] - a['score'], b['adj_edge_jaccard'] - a['adj_edge_jaccard'],
                                                     b['division_tp'] - a['division_tp'], b['division_fp'] - a['division_fp'])
        bs = [summarise([RR[j]['rows'][c] for j in k])['score'] - summarise([RR[j]['base'] for j in k])['score'] for k in K_]
        line += ' | CI [%+.5f,%+.5f]' % (np.quantile(bs, .025), np.quantile(bs, .975))
        cl = [r for r in RR if r['set'] in ('hold36', 'prev4')]
        line += ' | c40 %+.5f' % (summarise([r['rows'][c] for r in cl])['score'] - summarise([r['base'] for r in cl])['score'])
        print(line)
        for emb in ['44b6', '6bba']:
            lab = Counter()
            for r in RR:
                if r['movie'].startswith(emb): lab.update(r['lab'][c])
            dd = Counter()
            for r in RR:
                if r['movie'].startswith(emb):
                    for k in ['edge_tp', 'edge_fp', 'edge_fn', 'num_pred_nodes']: dd[k] += r['rows'][c][k] - r['base'][k]
            print('     %s labels %s | dTP %+d dFP %+d dFN %+d dN %+d' % (emb, dict(sorted(lab.items())), dd['edge_tp'], dd['edge_fp'], dd['edge_fn'], dd['num_pred_nodes']))
