"""Image-supported one-frame recovery with a learned parent-conditioned offspring map."""
from pathlib import Path
import os,json,time,hashlib
import numpy as np,torch
from scipy.spatial import cKDTree
from scipy.special import expit
from joint_model import JointMovie,load_joint,SHAPE,SPACING
from cell_event import SCALE,chain,edge_geometry,fork_geometry
from refine_events import structure

RECOVERY_DEFAULT={'heat_threshold':.60,'fork_threshold':.995,'edge_threshold':.995,'max_distance':14.,'min_separation':3.,'max_backtrack_error':5.,'min_history':3,'min_future':3,'allow_division':True,'allow_gap':True,'bridge_gap':1}

def bridge_future(q,d,out,pos,gap):
    if gap==1:return np.vstack([q,chain(d,out,pos)[:3]])
    return np.vstack([q+(pos[d]-q)*j/gap for j in range(gap)]+list(chain(d,out,pos)[:3]))[:4]
def local_heat_peak(heat,relative,axes,radius):
    # Restrict computation to the exact bounding box of the original spherical
    # search; axis values, scores, tie ordering and the selected voxel are unchanged.
    bounds=[(int(np.searchsorted(axis,r-radius,side='left')),int(np.searchsorted(axis,r+radius,side='right'))) for axis,r in zip(axes,relative)]
    if any(a>=b for a,b in bounds):return None
    slices=tuple(slice(a,b) for a,b in bounds);grid=np.stack(np.meshgrid(*[axis[a:b] for axis,(a,b) in zip(axes,bounds)],indexing='ij'),-1)
    distance=np.linalg.norm(grid-relative,axis=-1);allowed=distance<=radius
    if not allowed.any():return None
    scores=np.where(allowed,heat[slices]-.015*distance,-np.inf);local=np.unravel_index(np.argmax(scores),scores.shape)
    return tuple(int(i+a) for i,(a,b) in zip(local,bounds))

class RecoveryRefiner:
    def __init__(self,paths,root,config=None,cache_dir=None):
        self.paths=[Path(p) for p in paths];self.models=[load_joint(p)[0] for p in paths];self.root=Path(root);self.config={**RECOVERY_DEFAULT,**(config or {})}
        self.cache=Path(cache_dir or '/kaggle/working/recovery_cache');self.cache.mkdir(parents=True,exist_ok=True)
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    @torch.inference_mode()
    def proposals(self,name,nodes,edges):
        sha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16]
        cfg={k:self.config[k] for k in ['max_distance','min_separation','max_backtrack_error','min_history','min_future']}
        if self.config['bridge_gap']!=1:cfg['bridge_gap']=self.config['bridge_gap']
        cs=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]
        path=self.cache/(name+'_'+self.signature+'_'+sha+'_'+cs+'.json')
        if path.exists():return json.loads(path.read_text())
        out,prev,frames,pos=structure(nodes,edges);cfg=self.config;wanted={};gap=int(cfg['bridge_gap']);assert 1<=gap<=3
        frame_trees={t:cKDTree([pos[n] for n in ns]) for t,ns in frames.items() if ns}
        for t,sources in sorted(frames.items()):
            starts=[n for n in frames.get(t+gap+1,[]) if n not in prev and len(chain(n,out,pos))>=cfg['min_future']]
            if not starts or t+1 not in frames:continue
            tree=cKDTree([pos[n] for n in starts])
            for s in sources:
                old=out.get(s,[]);h=chain(s,prev,pos)
                if len(old)>1 or len(h)<cfg['min_history']:continue
                cur=s;recentfork=False
                for _ in range(4):
                    if cur not in prev:break
                    cur=prev[cur]
                    if len(out.get(cur,[]))>1:recentfork=True;break
                if old and (recentfork or len(chain(old[0],out,pos))<cfg['min_future']):continue
                candidates=[starts[j] for j in tree.query_ball_point(pos[s],cfg['max_distance']+4*gap)]
                candidates.sort(key=lambda d:np.linalg.norm(pos[d]-pos[s]));accepted=[]
                for d in candidates[:5]:
                    f=chain(d,out,pos);v=(f[-1]-f[0])/max(1,len(f)-1);expected=f[0]-v*gap
                    if np.linalg.norm(expected-pos[s])>cfg['max_distance'] or np.linalg.norm(v)>8:continue
                    accepted.append((d,expected))
                if accepted:wanted[s]=accepted
        if hasattr(self,'filter_wanted'):wanted=self.filter_wanted(nodes,edges,wanted)
        movie=JointMovie(self.root/(name+'.zarr'),max(m.context for m in self.models));sources=sorted(wanted,key=lambda s:(nodes[s]['t'],s));records=[]
        axes=[(np.arange(n)-n//2)*float(sp) for n,sp in zip(SHAPE,SPACING)];grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
        for start in range(0,len(sources),32):
            batch=sources[start:start+32];x=torch.from_numpy(movie.patches([nodes[s]['t'] for s in batch],[[nodes[s][k] for k in ['z','y','x']] for s in batch])).cuda().float()
            encoded=[]
            with torch.autocast('cuda',dtype=torch.float16):
                for m in self.models:encoded.append(m.encode(x))
            heat=np.mean([en[-2][:,0].float().sigmoid().cpu().numpy() for en in encoded],axis=0)
            requests=[]
            for bi,s in enumerate(batch):
                t=int(nodes[s]['t']);h=chain(s,prev,pos);old=out.get(s,[])
                for d,expected in wanted[s]:
                    rel=expected-pos[s];ix=local_heat_peak(heat[bi],rel,axes,cfg['max_backtrack_error'])
                    if ix is None:continue
                    q=pos[s]+grid[ix];hp=float(heat[bi][ix])
                    if hp<.20 or np.linalg.norm(q-pos[d])>9*gap or np.linalg.norm(q-pos[s])>cfg['max_distance']:continue
                    nearest,_=frame_trees[t+1].query(q)
                    if nearest<cfg['min_separation']:continue
                    xyz=q/SCALE
                    if np.any(xyz<0) or np.any(np.rint(xyz)>=np.asarray(movie.shape[1:])):continue
                    if gap>1 and any(t+j not in frame_trees or frame_trees[t+j].query(q+(pos[d]-q)*(j-1)/gap)[0]<cfg['min_separation'] for j in range(2,gap+1)):continue
                    ff=bridge_future(q,d,out,pos,gap)
                    if old:
                        a=old[0];relative=np.asarray([[0,0,0],pos[a]-pos[s],q-pos[s]],np.float32);geom=fork_geometry(h,chain(a,out,pos),ff);kind='fork'
                    else:relative=np.asarray([[0,0,0],q-pos[s]],np.float32);geom=edge_geometry(h,ff);kind='edge'
                    requests.append({'s':s,'d':d,'bi':bi,'position':q.tolist(),'heat':hp,'kind':kind,'relative':relative,'geometry':geom,'old':old})
            query_lists=[[] for _ in batch];query_maps=[{} for _ in batch]
            for r in requests:
                bi=r['bi'];r['query_indices']=[]
                for point in r['relative']:
                    key=tuple(map(float,point))
                    if key not in query_maps[bi]:query_maps[bi][key]=len(query_lists[bi]);query_lists[bi].append(point)
                    r['query_indices'].append(query_maps[bi][key])
            if not requests:continue
            coords=np.zeros((len(batch),max(map(len,query_lists)),3),np.float32)
            for bi,points in enumerate(query_lists):
                if points:coords[bi,:len(points)]=points
            active=sorted(set(r['bi'] for r in requests));active_lookup={bi:i for i,bi in enumerate(active)};active_ix=torch.tensor(active,device='cuda');qt=torch.from_numpy(coords[active]).cuda();pooled=[]
            for m,en in zip(self.models,encoded):
                unique_encoded=en if len(active)==len(batch) else tuple(e[active_ix] for e in en)
                with torch.autocast('cuda',dtype=torch.float16):pooled.append(m.pool(unique_encoded,qt))
            for kind in ['edge','fork']:
                chosen=[r for r in requests if r['kind']==kind]
                if not chosen:continue
                gg=torch.from_numpy(np.stack([r['geometry'] for r in chosen])).cuda();ix=torch.tensor([active_lookup[r['bi']] for r in chosen],device='cuda');qi=torch.tensor([r['query_indices'] for r in chosen],device='cuda');logs=[]
                for m,features in zip(self.models,pooled):
                    with torch.autocast('cuda',dtype=torch.float16):lp=m.classify_features(features[ix[:,None],qi],gg,kind)
                    logs.append(lp.float().cpu().numpy())
                probs=expit(np.mean(logs,axis=0))
                for r,p in zip(chosen,probs):
                    records.append({k:v for k,v in r.items() if k not in ['bi','relative','geometry','query_indices']}|{'probability':float(p)})
            if start%512==0:print('RECOVERY_PROPOSE',name,start,'/',len(sources),flush=True)
        tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(records));os.replace(tmp,path);return records
    def refine(self,name,nodes,edges):
        import zarr
        image_shape=np.asarray(zarr.open_group(str(self.root/(name+'.zarr')),mode='r')['0'].shape[1:])
        started=time.time();proposals=self.proposals(name,nodes,edges);cfg=self.config;newnodes=dict(nodes);newedges=[dict(e) for e in edges];useds=set();usedd=set();accepted=[];initial_max=max(nodes,default=-1);nid=initial_max+1
        for r in sorted(proposals,key=lambda r:r['probability']*r['heat'],reverse=True):
            s,d=r['s'],r['d'];kind=r['kind'];threshold=cfg['fork_threshold'] if kind=='fork' else cfg['edge_threshold']
            if r['heat']<cfg['heat_threshold'] or r['probability']<threshold or s in useds or d in usedd:continue
            if (kind=='fork' and not cfg['allow_division']) or (kind=='edge' and not cfg['allow_gap']):continue
            xyz=np.asarray(r['position'])/SCALE;t=int(nodes[s]['t'])+1
            if np.any(xyz<0) or np.any(np.rint(xyz)>=image_shape):continue
            if any(n['t']==t and np.linalg.norm((np.array([n[k] for k in ['z','y','x']])-xyz)*SCALE)<cfg['min_separation'] for i,n in newnodes.items() if i>initial_max):continue
            gap=int(cfg['bridge_gap']);destination=np.asarray([nodes[d][k] for k in ['z','y','x']]);coordinates=[xyz+(destination-xyz)*j/gap for j in range(gap)]
            if any(np.any(q<0) or np.any(np.rint(q)>=image_shape) for q in coordinates):continue
            if gap>1 and any(any(n['t']==t+j and np.linalg.norm((np.asarray([n[k] for k in ['z','y','x']])-q)*SCALE)<cfg['min_separation'] for i,n in newnodes.items() if i>initial_max) for j,q in enumerate(coordinates)):continue
            previous=s
            for j,q in enumerate(coordinates):
                newnodes[nid]={'node_id':nid,'t':t+j,**dict(zip(['z','y','x'],map(float,q))),'learned_recovery':1};newedges.append({'source_id':previous,'target_id':nid,'edge_prob':r['probability'],'learned_recovery':1});previous=nid;nid+=1
            newedges.append({'source_id':previous,'target_id':d,'edge_prob':r['probability'],'learned_recovery':1});accepted.append(r);useds.add(s);usedd.add(d)
        oo,pp,_,_=structure(newnodes,newedges);assert len(pp)==len(newedges) and all(len(ds)<=2 for ds in oo.values())
        assert all(newnodes[e['target_id']]['t']==newnodes[e['source_id']]['t']+1 for e in newedges)
        stats={'proposals':len(proposals),'recovered_nodes':len(accepted)*int(cfg['bridge_gap']),'recovered_divisions':sum(r['kind']=='fork' for r in accepted),'seconds':round(time.time()-started,2)};print('RECOVERY_REFINED',name,json.dumps(stats),flush=True);return newnodes,newedges,stats
