"""Learned recovery of a daughter observation immediately before a late split.

The operation requires a second image-supported offspring peak, supported future
tracks, and high learned mother/two-daughter probability. No annotations are read.
"""
from pathlib import Path
import os,json,time,hashlib
import numpy as np,torch
from scipy.spatial import cKDTree
from scipy.special import expit
from joint_model import JointMovie,load_joint,SHAPE,SPACING
from cell_event import SCALE,chain,fork_geometry
from refine_events import structure

RETIME_DEFAULT={'retime_probability':.99,'retime_heat':.60,'retime_separation':3.0,'retime_max_error':5.0}
class RetimingRefiner:
    def __init__(self,paths,root,config=None,cache_dir=None):
        self.paths=[Path(p) for p in paths];self.models=[load_joint(p)[0] for p in paths];self.root=Path(root);self.config={**RETIME_DEFAULT,**(config or {})};self.cache=Path(cache_dir or '/kaggle/working/retime_cache');self.cache.mkdir(parents=True,exist_ok=True)
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    @torch.inference_mode()
    def proposals(self,name,nodes,edges):
        sha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16];path=self.cache/('retime_'+name+'_'+self.signature+'_'+sha+'.json')
        if path.exists():return json.loads(path.read_text())
        out,prev,frames,pos=structure(nodes,edges);forks=[f for f,ds in out.items() if len(ds)==2 and f in prev and len(out.get(prev[f],[]))==1 and len(chain(prev[f],prev,pos))>=2 and all(len(chain(d,out,pos))>=3 for d in ds)]
        forks.sort(key=lambda f:(nodes[f]['t'],f));movie=JointMovie(self.root/(name+'.zarr'),max(m.context for m in self.models));trees={t:cKDTree([pos[n] for n in ns]) for t,ns in frames.items()};records=[]
        axes=[(np.arange(n)-n//2)*float(sp) for n,sp in zip(SHAPE,SPACING)];grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
        for f in forks:
            p=prev[f];t=int(nodes[p]['t']);h=chain(p,prev,pos);x=torch.from_numpy(movie.patch(t,[nodes[p][k] for k in ['z','y','x']])[None]).cuda().float();encoded=[]
            with torch.autocast('cuda',dtype=torch.float16):
                for model in self.models:encoded.append(model.encode(x))
            heat=np.mean([e[-2][0,0].float().sigmoid().cpu().numpy() for e in encoded],axis=0)
            for a,b in [out[f],list(reversed(out[f]))]:
                aa=chain(a,out,pos);bb=chain(b,out,pos);v=(bb[-1]-bb[0])/max(1,len(bb)-1);expected=bb[0]-v
                if np.linalg.norm(pos[a]-pos[f])>9 or np.linalg.norm(expected-pos[p])>14:continue
                distance=np.linalg.norm(grid-(expected-pos[p]),axis=-1);allowed=distance<=5.;allowed&=np.linalg.norm(grid-(pos[f]-pos[p]),axis=-1)>=3.
                if not allowed.any():continue
                ix=np.unravel_index(np.argmax(np.where(allowed,heat-.015*distance,-np.inf)),SHAPE);q=pos[p]+grid[ix];hp=float(heat[ix]);nearest,_=trees[t+1].query(q)
                if hp<.25 or nearest<3. or np.linalg.norm(q-pos[b])>9:continue
                xyz=q/SCALE
                if np.any(xyz<0) or np.any(np.rint(xyz)>=np.asarray(movie.shape[1:])):continue
                first=np.vstack([pos[f],aa[:3]]);second=np.vstack([q,bb[:3]])
                relative=torch.from_numpy(np.asarray([[[0,0,0],pos[f]-pos[p],q-pos[p]]],np.float32)).cuda();g=torch.from_numpy(fork_geometry(h,first,second)[None]).cuda();logs=[]
                for model,en in zip(self.models,encoded):
                    with torch.autocast('cuda',dtype=torch.float16):logits=model.classify(en,relative,g,'fork')
                    logs.append(float(logits.float()))
                records.append({'parent':p,'fork':f,'stay':a,'moved':b,'position':q.tolist(),'heat':hp,'probability':float(expit(np.mean(logs))),'backtrack_error':float(np.linalg.norm(q-expected)),'separation':float(nearest)})
        tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(records));os.replace(tmp,path);return records
    def refine(self,name,nodes,edges):
        import zarr
        image_shape=np.asarray(zarr.open_group(str(self.root/(name+'.zarr')),mode='r')['0'].shape[1:])
        start=time.time();cfg=self.config;proposals=self.proposals(name,nodes,edges);nn=dict(nodes);ee=[dict(e) for e in edges];used=set();nid=max(nodes,default=-1)+1;count=0
        for r in sorted(proposals,key=lambda r:r['probability']*r['heat'],reverse=True):
            if r['fork'] in used or r['parent'] in used or r['probability']<cfg['retime_probability'] or r['heat']<cfg['retime_heat'] or r['backtrack_error']>cfg['retime_max_error'] or r['separation']<cfg['retime_separation']:continue
            p,f,b=r['parent'],r['fork'],r['moved'];xyz=np.asarray(r['position'])/SCALE;t=int(nodes[f]['t'])
            if np.any(xyz<0) or np.any(np.rint(xyz)>=image_shape):continue
            nn[nid]={'node_id':nid,'t':t,**dict(zip(['z','y','x'],map(float,xyz))),'learned_retiming':1}
            ee=[e for e in ee if (int(e['source_id']),int(e['target_id']))!=(f,b)]
            ee.extend([{'source_id':p,'target_id':nid,'edge_prob':r['probability'],'learned_retiming':1},{'source_id':nid,'target_id':b,'edge_prob':r['probability'],'learned_retiming':1}]);used.update([p,f]);nid+=1;count+=1
        oo,pp,_,_=structure(nn,ee);assert len(pp)==len(ee) and all(len(ds)<=2 for ds in oo.values());assert all(nn[e['target_id']]['t']==nn[e['source_id']]['t']+1 for e in ee)
        stats={'retimed_divisions':count,'retiming_proposals':len(proposals),'retiming_seconds':round(time.time()-start,2)};print('DIVISION_RETIMING',name,json.dumps(stats),flush=True);return nn,ee,stats
