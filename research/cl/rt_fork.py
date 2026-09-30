import sys
sys.path.insert(0, '/workspace/cl/p16/redteam_p20')
from rt_common import *
rows = json.load(open('/workspace/cl/p16/redteam_p20/rt_a.json'))
for r in rows:
    if r['forks_lost'] or r['fork_break_rule']:
        print(r['set'], r['movie'], 'lost', r['forks_lost'], 'rule', r['fork_break_rule'], 'new', r['forks_new'])
        s, m = r['set'], r['movie']
        n15, e15, _ = load(p15p(s, m)); n19, e19, _ = load(p19p(s, m))
        F15 = forks(e15); F19 = forks(e19)
        out = defaultdict(list); par = {}
        fl = {}
        for ed in e15:
            a, b = int(ed['source_id']), int(ed['target_id']); out[a].append(b); par[b] = a
            fl[(a, b)] = {k: v for k, v in ed.items() if k not in ('source_id', 'target_id')}
        S = np.array([1.625, .40625, .40625])
        pos = lambda x: np.array([max(0, int(round(float(n15[x][k])))) for k in 'zyx']) * S
        for p in r['forks_lost']:
            c = F15[p]
            print(' parent', p, 't', n15[p]['t'], 'children', c, 'in19: parent', p in n19, 'children', [x in n19 for x in c], 'F19', F19.get(p))
            # show neighborhood: chain back 5 from parent and forward 6 from children
            for x in c:
                ch = [x]
                while len(out.get(ch[-1], [])) == 1 and len(ch) < 12: ch.append(out[ch[-1]][0])
                print('  daughter chain', [(y, int(n15[y]['t']), y in n19) for y in ch], 'edge attrs', fl[(p, x)])
            back = [p]
            while back[-1] in par and len(back) < 8: back.append(par[back[-1]])
            print('  parent chain back', [(y, int(n15[y]['t']), y in n19, len(out.get(y, []))) for y in back])
            # deleted nodes near
            dels = [y for y in set(n15) - set(n19) if abs(int(n15[y]['t']) - int(n15[p]['t'])) <= 8]
            print('  deleted near t:', [(y, int(n15[y]['t']), round(float(min(np.linalg.norm(pos(y) - pos(z)) for z in (p,) + c)), 2), 'par', par.get(y), 'out', out.get(y)) for y in sorted(dels, key=lambda y: int(n15[y]['t']))][:20])
            # nearest other node distances of daughters/parent in same frame
            for z in (p,) + c:
                tz = int(n15[z]['t'])
                same = [y for y in n15 if int(n15[y]['t']) == tz and y != z]
                d = sorted((float(np.linalg.norm(pos(y) - pos(z))), y) for y in same)[:3]
                print('  NN of', z, [(round(a, 2), y, par.get(y), out.get(y)) for a, y in d])
