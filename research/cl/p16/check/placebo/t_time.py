import os, sys, time
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']: os.environ[k]='1'
sys.path.insert(0,'/workspace/cl'); sys.path.insert(0,'/workspace/official/src'); sys.path.insert(0,'/workspace/code')
import evalx, glob
f = sorted(glob.glob('/workspace/cl/ps_p15_hold36/graphs/*.json'))[0]
t=time.time(); nodes, edges = evalx.load_graph_json(f); t1=time.time()
gt, nt = evalx.load_gt(os.path.basename(f)[:-5]); t2=time.time()
r = evalx.score_movie(os.path.basename(f)[:-5], nodes, edges); t3=time.time()
print(len(nodes), len(edges), gt.num_nodes(), gt.num_edges(), 'load %.1f gt %.1f score %.1f'%(t1-t, t2-t1, t3-t2))
print(r)
