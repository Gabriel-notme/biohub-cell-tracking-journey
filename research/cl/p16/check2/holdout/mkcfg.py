import json
DEF = dict(cd=3.2, ff=6, st=2.5, tt=3.5, par=3.5, bm=2.0, bl=6)
GRID = {'cd': [None, 2.0, 2.6, 3.8, 4.5, 5.5],
        'ff': [None, 3, 4, 5, 7, 8, 10],
        'st': [None, 1.5, 2.0, 3.0, 3.5, 4.5],
        'tt': [None, 2.5, 3.0, 4.0, 4.5],
        'par': [None, 2.5, 3.0, 4.0, 4.5, 5.5],
        'bm': [None, 0.5, 1.0, 1.5, 3.0, 4.0, 6.0],
        'bl': [3, 4, 5, 7, 8, 10]}
cfgs = [dict(DEF)] + [dict(DEF, **{k: v}) for k, vs in GRID.items() for v in vs]
OFF = dict(cd=None, ff=None, st=None, tt=None, par=None, bm=None, bl=6)
cfgs += [dict(OFF, cd=3.2), dict(OFF, ff=6), dict(OFF, st=2.5, tt=3.5, par=3.5), dict(OFF, bm=2.0), dict(OFF, cd=3.2, ff=6),
         dict(OFF, st=2.5), dict(OFF, tt=3.5), dict(OFF, par=3.5)]
json.dump(cfgs, open('/workspace/cl/p16/check2/holdout/cfgs1.json', 'w'))
print(len(cfgs))
