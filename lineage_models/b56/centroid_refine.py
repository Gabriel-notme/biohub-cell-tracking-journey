from pathlib import Path
import os,json,hashlib,time
import numpy as np,torch
from scipy.spatial import cKDTree
from centroid_model import load_centroid
from cell_event import Movie,SCALE,STRIDE
from refine_events import structure
DEFAULT={'centroid_blend':.5,'centroid_max_shift':3.,'centroid_max_error':2.5,'centroid_min_separation':2.}
class CentroidRefiner:
    def __init__(self,paths,root,config=None,cache_dir=None):
        assert 1<=len(paths)<=2
        self.paths=[Path(p) for p in paths];self.models=[load_centroid(p)[0] for p in paths];self.root=Path(root);self.config={**DEFAULT,**(config or {})};self.cache=Path(cache_dir);self.cache.mkdir(parents=True,exist_ok=True);self.signature=hashlib.sha256(self.paths[0].read_bytes()).hexdigest()[:16] if len(paths)==1 else hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    @torch.inference_mode()
    def predictions(self,name,nodes):
        sha=hashlib.sha256(json.dumps(nodes,sort_keys=True).encode()).hexdigest()[:16];path=self.cache/('centroid_v1_'+name+'_'+self.signature+'_'+sha+'.npz');ids=sorted(nodes,key=lambda n:(nodes[n]['t'],n))
        if path.exists():
            with np.load(path) as d:return ids,d['delta'],d['error']
        movie=Movie(self.root/(name+'.zarr'),7);delta=[];error=[]
        for start in range(0,len(ids),256):
            ns=ids[start:start+256];coords=np.asarray([[nodes[n][k] for k in ['z','y','x']] for n in ns]);fraction=(coords-np.rint(coords/STRIDE)*STRIDE)*SCALE;x=torch.from_numpy(movie.patches([nodes[n]['t'] for n in ns],coords)).cuda().float();f=torch.from_numpy(fraction.astype(np.float32)).cuda()
            with torch.autocast('cuda',dtype=torch.float16):predictions=[m(x,f) for m in self.models]
            if len(predictions)==1:dd,ee=predictions[0]
            else:dd=torch.stack([p[0].float() for p in predictions]).mean(0);ee=torch.stack([p[1].float() for p in predictions]).mean(0)
            delta.append(dd.float().cpu().numpy());error.append(ee.float().cpu().numpy())
        delta=np.concatenate(delta);error=np.concatenate(error);tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,delta=delta,error=error);os.replace(tmp,path);return ids,delta,error
    def refine(self,name,nodes,edges):
        import zarr
        start=time.time();ids,delta,error=self.predictions(name,nodes);cfg=self.config;_,_,frames,pos=structure(nodes,edges);shape=np.asarray(zarr.open_group(str(self.root/(name+'.zarr')),mode='r')['0'].shape[1:]);result={n:dict(v) for n,v in nodes.items()};changed=set()
        for n,d,e in zip(ids,delta,error):
            if e>cfg['centroid_max_error']:continue
            length=float(np.linalg.norm(d));d=d*min(1.,cfg['centroid_max_shift']/max(length,1e-8));q=(pos[n]+cfg['centroid_blend']*d)/SCALE
            if np.any(q<0) or np.any(np.rint(q)>=shape):continue
            result[n].update(dict(zip(['z','y','x'],map(float,q))));changed.add(n)
        # Reject movements that create a new sub-resolution collision.
        for ns in frames.values():
            old=np.asarray([pos[n] for n in ns])
            for _ in range(3):
                pp=np.asarray([[result[n][k] for k in ['z','y','x']] for n in ns])*SCALE;rejected=set()
                for i,j in cKDTree(pp).query_pairs(cfg['centroid_min_separation']):
                    if np.linalg.norm(pp[i]-pp[j])+1e-5<min(cfg['centroid_min_separation'],np.linalg.norm(old[i]-old[j])):rejected.update(n for n in [ns[i],ns[j]] if n in changed)
                if not rejected:break
                for n in rejected:result[n]=dict(nodes[n]);changed.discard(n)
        stats={'relocated_nodes':len(changed),'seconds':time.time()-start};print('CENTROID_REFINED',name,json.dumps(stats),flush=True);return result,edges,stats
