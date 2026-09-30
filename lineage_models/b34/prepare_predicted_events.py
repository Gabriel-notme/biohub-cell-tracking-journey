"""Seven-frame image training at actual predicted centroids and graph histories."""
from pathlib import Path
import os,json,time,argparse,multiprocessing as mp
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2')
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import linear_sum_assignment
from prepare_events import graph
from refine_events import structure
from joint_refine import JointRefiner,JOINT_DEFAULT
from cell_event import Movie,SCALE,chain,edge_geometry,fork_geometry
R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train');DEST=R/'predicted_events'

def prepare(name,group):
    started=time.time();marker=DEST/(name+'_ready.json')
    if marker.exists():return json.loads(marker.read_text())
    folder='training_baseline_graphs' if group=='train' else 'prior_graphs_B2';d=json.loads((R/folder/(name+'.json')).read_text());nodes={int(k):v for k,v in d['nodes'].items()};edges=d['edges'];out,prev,frames,pos=structure(nodes,edges)
    gn,go,gp,gf,gpos=graph(TRAIN/(name+'.geff'));matched={}
    for t,gs in gf.items():
        ns=frames.get(t,[])
        if not ns:continue
        distance=np.linalg.norm(np.asarray([gpos[g] for g in gs])[:,None]-np.asarray([pos[n] for n in ns])[None],axis=-1);ii,jj=linear_sum_assignment(distance)
        for i,j in zip(ii,jj):
            if distance[i,j]<=6:matched[ns[j]]=gs[i]
    extractor=JointRefiner.__new__(JointRefiner);extractor.config=dict(JOINT_DEFAULT);er,fr,eg,fg=extractor.candidates(nodes,edges)
    requests=[];lookup={};phase=[];rng=np.random.default_rng(int.from_bytes(name.encode()[-5:],'little')%2**32)
    def request(t,p,label=-1):
        coord=p/SCALE;key=(int(t),*np.rint(coord*32).astype(int).tolist())
        if key not in lookup:lookup[key]=len(requests);requests.append((int(t),coord));phase.append(label)
        return lookup[key]
    data={}
    for kind,candidates,geom in [('edge',er,eg),('fork',fr,fg)]:
        chosen=[]
        for row,g in zip(candidates,geom):
            s,*targets=map(int,row)
            if s not in matched or matched[s] not in go:continue
            gs=matched[s];ds=[];ambiguous=False
            for target in targets:
                if target in matched:ds.append(matched[target])
                else:
                    if min(np.linalg.norm(pos[target]-gpos[k]) for k in go[gs])<=7:ambiguous=True;break
                    ds.append(-1)
            if ambiguous:continue
            y=int(ds[0] in go[gs]) if kind=='edge' else int(set(ds)==set(go[gs]) and len(go[gs])==2)
            chosen.append((row,g,y))
        positive=[r for r in chosen if r[-1]];negative=[r for r in chosen if not r[-1]]
        hard_sources={int(r[0][0]) for r in positive};hard=[r for r in negative if int(r[0][0]) in hard_sources];ordinary=[r for r in negative if int(r[0][0]) not in hard_sources]
        cap=2000 if kind=='edge' else 1200
        if len(ordinary)>cap:ordinary=[ordinary[i] for i in rng.choice(len(ordinary),cap,replace=False)]
        selected=positive+hard+ordinary;rows=[];geometry=[]
        for row,g,y in selected:
            ids=[request(nodes[int(n)]['t'],pos[int(n)],int(len(go.get(matched[int(n)],[]))==2) if int(n) in matched else -1) for n in row]
            rows.append([*ids,y]);geometry.append(g)
        # Ground-truth-centered positive divisions supplement merged/absent
        # detections only on training movies, never on calibration or audit.
        if kind=='fork' and group=='train':
            for s,ds in go.items():
                if len(ds)!=2:continue
                ids=[request(gn[n]['t'],gpos[n],int(len(go.get(n,[]))==2)) for n in [s,*ds]]
                rows.append([*ids,1]);geometry.append(fork_geometry(chain(s,gp,gpos),chain(ds[0],go,gpos),chain(ds[1],go,gpos)))
        data['edges' if kind=='edge' else 'forks']=np.asarray(rows,np.int32).reshape(-1,3 if kind=='edge' else 4);data[kind+'_geom']=np.asarray(geometry,np.float32).reshape(-1,16 if kind=='edge' else 28)
    data['phase']=np.asarray(phase,np.float32);data['node_ids']=np.arange(len(requests));movie=Movie(TRAIN/(name+'.zarr'),context=7);bank=np.lib.format.open_memmap(DEST/(name+'_patches.npy'),mode='w+',dtype=np.float16,shape=(len(requests),7,12,24,24))
    for i in sorted(range(len(requests)),key=lambda k:requests[k][0]):bank[i]=movie.patch(*requests[i])
    bank.flush();del bank;tmp=DEST/(name+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**data);os.replace(tmp,DEST/(name+'.npz'))
    report={'movie':name,'group':group,'queries':len(requests),'edge_n':len(data['edges']),'fork_n':len(data['forks']),'edge_positive':int(data['edges'][:,-1].sum()),'fork_positive':int(data['forks'][:,-1].sum()),'seconds':time.time()-started};marker.write_text(json.dumps(report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args();DEST.mkdir(exist_ok=True);split=json.loads((R/'events/split.json').read_text());(DEST/'split.json').write_text(json.dumps(split,indent=2));pending={n:g for g in ['train','calibration'] for n in split[g]}
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        jobs={}
        while pending or jobs:
            for name,group in list(pending.items()):
                source=R/('training_baseline_graphs' if group=='train' else 'prior_graphs_B2')/(name+'.json')
                if source.exists() and len(jobs)<a.workers*2:jobs[pool.submit(prepare,name,group)]=name;del pending[name]
            for f in [f for f in jobs if f.done()]:
                print('PREDICTED_EVENTS_PREPARED',json.dumps(f.result()),flush=True);del jobs[f]
            time.sleep(3)
    (DEST/'ready.json').write_text(json.dumps({'ready':True,'movies':sum(len(split[g]) for g in ['train','calibration'])}))
