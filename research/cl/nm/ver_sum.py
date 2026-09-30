import json, glob
from collections import Counter, defaultdict
R = json.load(open('/workspace/cl/nm/ver_census.json'))
for emb in ['44b6', '6bba']:
    fn = Counter(); uc = Counter(); ts = Counter()
    for r in R:
        if r['emb'] != emb: continue
        fn.update(r['fn']); uc.update(r['ucov']); ts.update(r['tstat'])
    print('=====', emb, 'movies', sum(1 for r in R if r['emb'] == emb))
    print(' GT edge census', dict(sorted(fn.items())))
    print(' unmatched GT node cover', dict(uc))
    print(' termini', {k: v for k, v in sorted(ts.items()) if not k.startswith('unm')}, ' unm:', {k: v for k, v in ts.items() if k.startswith('unm')})


def pb(p):
    for lo in [0.8, 0.5, 0.3, 0.1]:
        if p >= lo: return '>=%.1f' % lo
    return '<0.1'


W = defaultdict(list)
for f in glob.glob('/workspace/cl/nm/ver_cands/*/*.json'):
    d = json.load(open(f))
    W[d['movie'][:4]].append(d)
for emb in ['44b6', '6bba', 'all']:
    Ds = W['44b6'] + W['6bba'] if emb == 'all' else W[emb]
    st = defaultdict(Counter); jn = defaultdict(Counter); nr = defaultdict(Counter); orc = Counter()
    for d in Ds:
        for w in d['walks']:
            prefix_tp = True
            for k, s in enumerate(w['steps']):
                key = ('first' if k == 0 else 'later', pb(s['p']), 'dupOK')
                st[key][s['lab']] += 1
                st[('ALL', pb(s['p']))][s['lab']] += 1
                st[('ilpE%d' % s['ilpe'], pb(s['p']))][s['lab']] += 1
                if prefix_tp and s['lab'] == 'TP': orc['steps'] += 1
                else: prefix_tp = False
                orc['n_steps'] += 1
            if w['join']:
                j = w['join']; jn[pb(j['p'])][j['lab']] += 1
                if prefix_tp and j['lab'] == 'TP': orc['joins'] += 1
            if w['near']:
                nr['pedge%d' % w['near']['pedge']][w['near']['lab']] += 1
                if w['near']['lab'] == 'TP': orc['near'] += 1
            if w['steps'] and w['steps'][0]['lab'] != 'NE': orc['walks_eval_first'] += 1
    print('=====', emb, 'oracle TP-prefix counts', dict(orc))
    for k in sorted(st, key=str):
        c = st[k]; ev = c['TP'] + c['FP']
        print('  step %-28s TP %5d FP %5d NE %6d | eval prec %s' % (' '.join(k), c['TP'], c['FP'], c['NE'], '%.2f' % (c['TP'] / ev) if ev else '-'))
    for k in sorted(jn):
        c = jn[k]; ev = c['TP'] + c['FP']
        print('  join %-10s TP %5d FP %5d NE %6d | eval prec %s' % (k, c['TP'], c['FP'], c['NE'], '%.2f' % (c['TP'] / ev) if ev else '-'))
    for k in sorted(nr):
        c = nr[k]; ev = c['TP'] + c['FP']
        print('  near %-10s TP %5d FP %5d NE %6d | eval prec %s' % (k, c['TP'], c['FP'], c['NE'], '%.2f' % (c['TP'] / ev) if ev else '-'))
