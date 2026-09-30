import sys, time
sys.path.insert(0,'/workspace/cl'); sys.path.insert(0,'/workspace/official/src'); sys.path.insert(0,'/workspace/code')
import evalx
t=time.time()
n,e=evalx.load_graph_json('/workspace/cl/ps_p15_hold36/graphs/44b6_144b256d.json')
r=evalx.score_movie('44b6_144b256d',n,e); print('ROW', r, time.time()-t)
