import json
c = json.load(open('/workspace/p19ds/p15_config.json'))
c['variant'] = 'P19R_chk2_reimpl'
c['p17'] = {'cutdup_on': 1, 'forkfrag_on': 1, 'shortbranch_on': 0}
c['p19small'] = {'dup': 1, 'border': 1}
json.dump(c, open('/workspace/cl/p16/check2/reimpl/cfg_p19r_chk.json', 'w'), indent=1)
r = json.load(open('/workspace/p19ds/p19r_config.json'))
print('equal to p19r_config except variant:', {k: v for k, v in c.items() if k != 'variant'} == {k: v for k, v in r.items() if k != 'variant'})
# control: plain P15 config through the same p_stage12.py (to separate P-stage reproducibility from the p17/p19 blocks)
c15 = json.load(open('/workspace/p19ds/p15_config.json')); c15['variant'] = 'P15_chk2_reimpl'
json.dump(c15, open('/workspace/cl/p16/check2/reimpl/cfg_p15_chk.json', 'w'), indent=1)

