"""Train-only graph corruption teaches recovery from incorrect tracklet context."""
from pathlib import Path
import os,json,hashlib,time
os.environ.update(TRACK_PREDICTED_FOLDER='track_corrupt3',TRACK_VIDEO_CHANNELS='3',TRACK_GRAPH_STAGE='B4',TRACK_MAX_CANDIDATES='8',TRACK_MAX_DISTANCE='18',TRACK_TRAIN_GRAPH_FOLDER='corrupted_training_graphs')
import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from prepare_track_predicted import prepare
from refine_events import structure
R=Path('/workspace/biohub');O=R/'track_corrupt3';G=R/'corrupted_training_graphs';D=R/'track_b4w_3'

def one(name):
    out=O/('aug__'+name+'.npz')
    if out.exists():return {'movie':name,'cached':True}
    raw=json.loads((R/'b4_training_graphs'/(name+'.json')).read_text());nodes={int(k):v for k,v in raw['nodes'].items()}
    outgoing,prev,frames,pos=structure(nodes,raw['edges']);edges={(int(e['source_id']),int(e['target_id'])):dict(e) for e in raw['edges']}
    rng=np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8],16));stats={'swaps':0,'false_forks':0,'breaks':0}
    touched=set()
    for t,ns in frames.items():
        sources=[s for s in ns if len(outgoing.get(s,[]))==1 and len(outgoing.get(prev.get(s,-1),[]))<=1]
        targets=frames.get(t+1,[])
        if not targets:continue
        tree=cKDTree([pos[d] for d in targets])
        rng.shuffle(sources)
        for s in sources:
            if s in touched or rng.random()>.08:continue
            a=outgoing[s][0];near=[targets[i] for i in tree.query_ball_point(pos[a],8.)]
            near=[d for d in near if d!=a and d in prev and prev[d] not in touched and len(outgoing.get(prev[d],[]))==1]
            if not near:continue
            d=near[int(rng.integers(len(near)))];owner=prev[d]
            if (s,a) not in edges or (owner,d) not in edges:continue
            u=rng.random()
            if u<.5:
                del edges[(s,a)];del edges[(owner,d)]
                edges[(s,d)]={'source_id':s,'target_id':d,'edge_prob':.95};edges[(owner,a)]={'source_id':owner,'target_id':a,'edge_prob':.95};stats['swaps']+=1
            elif u<.8:
                del edges[(owner,d)];edges[(s,d)]={'source_id':s,'target_id':d,'edge_prob':.95};stats['false_forks']+=1
            else:del edges[(s,a)];stats['breaks']+=1
            touched.update([s,owner,a,d])
    raw['edges']=list(edges.values());oo,pp,_,_=structure(nodes,raw['edges']);assert len(pp)==len(raw['edges']) and max(map(len,oo.values()),default=0)<=2
    (G/(name+'.json')).write_text(json.dumps(raw));result=prepare(name,'train')
    (O/(name+'.npz')).rename(out);(O/(name+'_crops.npy')).rename(O/('aug__'+name+'_crops.npy'))
    for suffix in ['.npz','_crops.npy']:(O/(name+suffix)).symlink_to(D/(name+suffix))
    return {'movie':name,**stats,'bank':result}

if __name__=='__main__':
    O.mkdir(exist_ok=True);G.mkdir(exist_ok=True);split=json.loads((D/'split.json').read_text());stats=[]
    for name in split['calibration']:
        for suffix in ['.npz','_crops.npy']:
            p=O/(name+suffix)
            if not p.exists():p.symlink_to(D/(name+suffix))
    with ProcessPoolExecutor(max_workers=8) as pool:
        for row in pool.map(one,split['train']):stats.append(row);print('CORRUPTION_BANK',json.dumps(row),flush=True)
    original=list(split['train']);split['train']+=['aug__'+n for n in original]
    split['augmentation_source_movies']={('aug__'+n):n for n in original}
    (O/'split.json').write_text(json.dumps(split));(O/'corruption_protocol.json').write_text(json.dumps({'train_only':True,'fitting_source_movies':original,'corruptions':'8% sampled ordinary sources: local child swaps, false forks and dropped links; annotations unchanged; clean and corrupted banks mixed 1:1','statistics':stats},indent=2))
    (O/'ready.json').write_text(json.dumps({'training_banks':len(split['train'])}));print('CORRUPTION_DATA_READY',flush=True)
