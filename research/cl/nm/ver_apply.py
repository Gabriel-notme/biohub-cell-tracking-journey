"""ver_apply: apply candidate additions enumerated by ver_cands.py on top of P15, with a selection rule.
fam : 'walk' (pre-ILP edge walks + joins), 'near' (nearest dropped detection without pre-ILP step), 'both'
sel : 'oracle' (GT: accept a walk prefix of TP steps, TP joins, TP near)   [AUDIT ONLY]
      'noFP'   (GT: accept until the first FP step; NE accepted)          [AUDIT ONLY]
      'all'    (accept every step with pmin <= p < pmax, prefix semantics)
      'score'  (per-candidate score from JSON file {movie: {key: score}}, accept if >= th; prefix semantics)
first_only: only the first step of each walk. kmax: max steps per walk."""
WANTS_META = True
import json


def apply(nodes, edges, fam='walk', sel='all', pmin=0.0, pmax=1.01, kmax=12, join=True, first_only=False, score=None, th=0.5,
          jmin=None, nearmax=6.0, th_emb=None, **meta):
    if th_emb is not None: th = th_emb[meta['name'][:4]]
    path = '/workspace/cl/nm/ver_cands/%s/%s.json' % (meta['set'], meta['name'])
    W = json.load(open(path))['walks']
    S = None
    if sel == 'score':
        S = _load(score).get(meta['name'], {})
    par = set(); chi = set()
    for e in edges: chi.add(int(e['source_id'])); par.add(int(e['target_id']))
    nn = dict(nodes); ne = list(edges); st = {'v_steps': 0, 'v_join': 0, 'v_near': 0}

    def ok(key, c, is_step=True):
        if sel == 'oracle': return c['lab'] == 'TP'
        if sel == 'noFP': return c['lab'] != 'FP'
        if sel == 'all': return (pmin <= c.get('p', 1.0) < pmax) if is_step else True
        if sel == 'score': return S.get(key, -1) >= th
        raise ValueError(sel)
    for w in W:
        n = int(w['seed']); fwd = bool(w['fwd'])
        if fwd and n in chi: continue
        if (not fwd) and n in par: continue
        cur = n; allok = True
        if fam in ('walk', 'both') and w['steps']:
            for k, s in enumerate(w['steps']):
                if k >= kmax or (first_only and k > 0): allok = False; break
                x = int(s['x'])
                if x in nn or not ok('%d_%d_s%d' % (n, int(fwd), k), s): allok = False; break
                nn[x] = {'node_id': x, 't': int(s['t']), 'z': s['z'], 'y': s['y'], 'x': s['xx'], 'ver_add': 1}
                if fwd: ne.append({'source_id': cur, 'target_id': x, 'ver_add': 1}); chi.add(cur); par.add(x)
                else: ne.append({'source_id': x, 'target_id': cur, 'ver_add': 1}); par.add(cur); chi.add(x)
                cur = x; st['v_steps'] += 1
            j = w.get('join')
            if join and allok and j is not None and len(w['steps']) <= kmax and not first_only:
                y = int(j['y'])
                jok = ok('%d_%d_j' % (n, int(fwd)), j, False) and (jmin is None or j['p'] >= jmin)
                if jok and y in nn:
                    if fwd and y not in par: ne.append({'source_id': cur, 'target_id': y, 'ver_join': 1}); par.add(y); chi.add(cur); st['v_join'] += 1
                    if (not fwd) and y not in chi: ne.append({'source_id': y, 'target_id': cur, 'ver_join': 1}); chi.add(y); par.add(cur); st['v_join'] += 1
        if fam in ('near', 'both') and w.get('near') is not None and not w['steps']:
            s = w['near']; x = int(s['x'])
            if x not in nn and s['dist'] <= nearmax and ok('%d_%d_n' % (n, int(fwd)), s, False):
                nn[x] = {'node_id': x, 't': int(s['t']), 'z': s.get('z'), 'y': s.get('y'), 'x': s.get('xx'), 'ver_near': 1}
                if nn[x]['z'] is None: del nn[x]; continue
                if fwd: ne.append({'source_id': n, 'target_id': x, 'ver_near': 1}); chi.add(n); par.add(x)
                else: ne.append({'source_id': x, 'target_id': n, 'ver_near': 1}); par.add(n); chi.add(x)
                st['v_near'] += 1
    return nn, ne, st


_C = {}


def _load(p):
    if p not in _C: _C[p] = json.load(open(p))
    return _C[p]
