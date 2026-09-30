"""Bounded, image-predicted centroid refinement; no labels or test-specific rules."""
from pathlib import Path
import os,json,hashlib,time
import numpy as np,torch
from scipy.spatial import cKDTree
from joint_model import JointMovie,load_joint,SHAPE,SPACING
from recover_joint import local_heat_peak
from refine_events import structure
from cell_event import SCALE,chain

class CoordinateRefiner:
    def __init__(self,paths,root,config=None,cache_dir=None):
        self.paths=[Path(p) for p in paths];self.models=[load_joint(p)[0] for p in paths];self.root=Path(root);self.config={'coordinate_radius':2.5,'coordinate_blend':.5,'coordinate_peak_margin':-2.,'coordinate_min_separation':2.,**(config or {})};self.cache=Path(cache_dir);self.cache.mkdir(exist_ok=True,parents=True);self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    @torch.inference_mode()
    def proposals(self,name,nodes,edges):
        cfg=self.config;identity=hashlib.sha256(json.dumps([nodes,edges,cfg['coordinate_radius']],sort_keys=True).encode()).hexdigest()[:16];path=self.cache/('coordinate_v1_'+name+'_'+self.signature+'_'+identity+'.json')
        if path.exists():return json.loads(path.read_text())
        out,prev,frames,pos=structure(nodes,edges);sources=sorted([s for s,ds in out.items() if len(chain(s,prev,pos))>=3 and all(np.linalg.norm(pos[d]-pos[s])<=14 for d in ds)],key=lambda s:(nodes[s]['t'],s));movie=JointMovie(self.root/(name+'.zarr'),7);axes=[(np.arange(n)-n//2)*float(sp) for n,sp in zip(SHAPE,SPACING)];grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1);records=[]
        for start in range(0,len(sources),32):
            batch=sources[start:start+32];x=torch.from_numpy(movie.patches([nodes[s]['t'] for s in batch],[[nodes[s][k] for k in ['z','y','x']] for s in batch])).cuda().float();heats=[]
            for model in self.models:
                with torch.autocast('cuda',dtype=torch.float16):heat=model.encode(x)[-2][:,0]
                heats.append(heat.float().cpu().numpy())
            heat=np.mean(heats,0)
            for bi,s in enumerate(batch):
                for d in out[s]:
                    relative=pos[d]-pos[s];ix=local_heat_peak(heat[bi],relative,axes,cfg['coordinate_radius'])
                    if ix is None:continue
                    # Fit a local soft centroid, preserving the learned spatial likelihood.
                    slices=tuple(slice(max(0,i-1),min(n,i+2)) for i,n in zip(ix,SHAPE));hh=heat[bi][slices];gg=grid[slices];weight=np.exp(hh-hh.max());q=(gg*weight[...,None]).sum((0,1,2))/weight.sum()+pos[s];shift=float(np.linalg.norm(q-pos[d]));margin=float(heat[bi][ix]-heat[bi].max())
                    if shift>cfg['coordinate_radius']:continue
                    xyz=q/SCALE
                    if np.any(xyz<0) or np.any(np.rint(xyz)>=np.asarray(movie.shape[1:])):continue
                    records.append({'node':int(d),'position':q.tolist(),'shift':shift,'peak_margin':margin})
            if start%4096==0:print('COORDINATE_INFER',name,start,'/',len(sources),flush=True)
        tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(records));os.replace(tmp,path);return records
    def refine(self,name,nodes,edges):
        started=time.time();records=self.proposals(name,nodes,edges);_,_,frames,pos=structure(nodes,edges);trees={t:cKDTree([pos[n] for n in ns]) for t,ns in frames.items()};result={n:dict(v) for n,v in nodes.items()};moves=[];cfg=self.config
        for r in records:
            if r['peak_margin']<cfg['coordinate_peak_margin']:continue
            n=r['node'];t=int(nodes[n]['t']);q=pos[n]+cfg['coordinate_blend']*(np.asarray(r['position'])-pos[n]);tree=trees[t];dist,ii=tree.query(q,k=min(3,len(frames[t])));others=[float(dd) for dd,j in zip(np.atleast_1d(dist),np.atleast_1d(ii)) if frames[t][int(j)]!=n]
            if others and min(others)<cfg['coordinate_min_separation']:continue
            xyz=q/SCALE;result[n].update(dict(zip(['z','y','x'],map(float,xyz))));moves.append(float(np.linalg.norm(q-pos[n])))
        stats={'relocated_nodes':len(moves),'mean_shift_um':float(np.mean(moves)) if moves else 0.,'seconds':time.time()-started};print('COORDINATE_REFINED',name,json.dumps(stats),flush=True);return result,edges,stats
