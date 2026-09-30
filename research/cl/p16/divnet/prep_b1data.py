"""Prepare b1-recipe event data (artifact prepare_events.prepare_one, unchanged) for all 199 movies into /dev/shm/divnet_ev,
then write two LOEO data dirs (symlinks + split.json): train on one embryo (calibration = 6 movies of the same embryo)."""
import os, sys, json, glob, random
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','BLOSC_NTHREADS']: os.environ[k]='1'
ART='/workspace/art_b56/artifact_bundle'; sys.path.insert(0,ART)
from multiprocessing import get_context
DST='/dev/shm/divnet_ev'
def job(name):
    import numcodecs.blosc; numcodecs.blosc.use_threads=False
    import prepare_events
    try: return prepare_events.prepare_one(('/workspace/data/train',DST,name,'all',5))
    except Exception as e: return {'movie':name,'error':repr(e)[:300]}
if __name__=='__main__':
    names=sorted(os.path.basename(f)[:-5] for f in glob.glob('/workspace/data/train/*.geff'))
    print(len(names),flush=True)
    with get_context('spawn').Pool(64,maxtasksperchild=4) as p:
        R=p.map(job,names,chunksize=1)
    err=[r for r in R if 'error' in r]; print('errors',len(err),err[:3],flush=True)
    for emb in ['44b6','6bba']:
        d='/dev/shm/divnet_ev_tr%s'%emb; os.makedirs(d,exist_ok=True)
        mv=[n for n in names if n.startswith(emb)]; rng=random.Random(20260928); cal=sorted(rng.sample(mv,6))
        for n in mv:
            for suf in ['.npz','_patches.npy']:
                t=os.path.join(d,n+suf)
                if not os.path.exists(t): os.symlink(os.path.join(DST,n+suf),t)
        json.dump({'train':[n for n in mv if n not in cal],'calibration':cal,'audit':[]},open(d+'/split.json','w'))
        print(emb,len(mv),cal,flush=True)
    print('PREP_DONE',flush=True)
