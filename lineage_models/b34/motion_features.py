"""Graph-context features for learned association on actual predicted trajectories."""
from pathlib import Path
import hashlib,json,os
import numpy as np
from scipy.spatial import cKDTree
from refine_events import structure
from cell_event import chain,edge_geometry,fork_geometry
from trajectory_model import AppearanceMovie,track_nodes

VERSION='graph_motion_photo_v1'
def invariant_features(x,kind):
    """Retain distances, angles and appearance; remove absolute-axis directions."""
    x=x.copy();directions=np.asarray([0,1,2,4,5,6,7,8,9,10,11,12])
    if kind=='edge':x[:,directions]=0
    else:
        # The final 96 columns contain min/max of the first 48 edge features.
        x[:,x.shape[1]-96+directions]=0;x[:,x.shape[1]-48+directions]=0
    return x
class MotionFeatures:
    def __init__(self,name,nodes,edges,root,cache_dir):
        self.nodes=nodes;self.out,self.prev,self.frames,self.pos=structure(nodes,edges)
        self.original={(int(e['source_id']),int(e['target_id'])):e for e in edges}
        cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
        coordsha=hashlib.sha256(json.dumps(nodes,sort_keys=True).encode()).hexdigest()[:16]
        path=cache/('appearance_'+name+'_'+coordsha+'.npz');ids=sorted(nodes)
        if path.exists():
            with np.load(path) as d:self.photo={int(n):v for n,v in zip(d['ids'],d['features'])}
        else:
            movie=AppearanceMovie(Path(root)/(name+'.zarr'))
            self.photo=movie.descriptors({n:(nodes[n]['t'],[nodes[n][k] for k in ['z','y','x']]) for n in ids})
            tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,ids=np.asarray(ids),features=np.stack([self.photo[n] for n in ids]));os.replace(tmp,path)
        self.hist={n:track_nodes(n,self.prev,nodes,8) for n in nodes};self.future={n:track_nodes(n,self.out,nodes,8) for n in nodes}
        self.hpoints={n:np.asarray([self.pos[k] for k in v]) for n,v in self.hist.items()};self.fpoints={n:np.asarray([self.pos[k] for k in v]) for n,v in self.future.items()}
        self.local={};self.density={};self.flow_spread={}
        for t,ns in self.frames.items():
            p=np.asarray([self.pos[n] for n in ns]);tree=cKDTree(p);dist,ii=tree.query(p,k=min(13,len(ns)))
            if ii.ndim==1:dist=dist[:,None];ii=ii[:,None]
            v=np.asarray([(self.pos[self.out[n][0]]-self.pos[n]) if len(self.out.get(n,[]))==1 else np.full(3,np.nan) for n in ns])
            usable=np.isfinite(v).all(1)&(np.linalg.norm(v,axis=1)<12);globalv=np.median(v[usable],axis=0) if usable.any() else np.zeros(3)
            for j,n in enumerate(ns):
                inds=ii[j,1:];dd=dist[j,1:];good=usable[inds]&(dd<25);vv=v[inds[good]]
                self.local[n]=np.median(vv,axis=0) if len(vv)>=3 else globalv
                self.flow_spread[n]=float(np.median(np.linalg.norm(vv-self.local[n],axis=1))) if len(vv) else 0.
                self.density[n]=np.asarray([float(dd[min(k,len(dd)-1)]) if len(dd) else 25. for k in [0,1,3,7]],np.float32)
        self.summaries={}
    def summary(self,n,forward):
        key=(n,forward)
        if key in self.summaries:return self.summaries[key]
        ids=(self.future if forward else self.hist)[n];points=np.asarray([self.pos[k] for k in ids]);photo=np.stack([self.photo[k] for k in ids]);delta=np.diff(points,axis=0)
        speed=np.linalg.norm(delta,axis=1) if len(delta) else np.zeros(1);acc=np.linalg.norm(np.diff(delta,axis=0),axis=1) if len(delta)>1 else np.zeros(1)
        f=np.r_[photo[0],photo[:3].mean(0),photo.mean(0),photo.std(0),photo[-1]-photo[0],len(ids)/8,speed.mean()/10,speed.max()/10,speed.std()/10,acc.mean()/10,acc.max()/10,self.density[n]/20,self.flow_spread[n]/10]
        self.summaries[key]=f.astype(np.float32);return self.summaries[key]
    @staticmethod
    def velocity(p,k,backward=False):
        j=min(k,len(p)-1);return ((p[0]-p[j]) if backward else (p[j]-p[0]))/max(j,1)
    def edge_one(self,s,d):
        p,q=self.pos[s],self.pos[d];delta=q-p;norm=np.linalg.norm(delta);h=self.hpoints[s];f=self.fpoints[d];a=self.summary(s,False);b=self.summary(d,True)
        features=list(edge_geometry(h[:4],f[:4]));features.extend([norm/10,np.linalg.norm(delta-self.local[s])/10,np.linalg.norm(delta-self.local[d])/10,np.linalg.norm(self.local[s]-self.local[d])/10])
        for k in [1,2,4,7]:
            v=self.velocity(h,k,True);w=self.velocity(f,k)
            features.extend([np.linalg.norm(delta-v)/10,np.linalg.norm(delta-w)/10,np.linalg.norm(v-w)/10,np.dot(v,w)/(np.linalg.norm(v)*np.linalg.norm(w)+1e-5),np.linalg.norm(v)/10,np.linalg.norm(w)/10])
        original=self.original.get((s,d),{}).get('edge_prob');features.extend([float(original) if original is not None else .5,float((s,d) in self.original),len(self.out.get(s,[])),int(d in self.prev)])
        return np.nan_to_num(np.r_[features,a,b,np.abs(a[:24]-b[:24]),self.density[d]-self.density[s]]).astype(np.float32)
    def fork_one(self,s,a,b):
        h=self.hpoints[s];fa=self.fpoints[a];fb=self.fpoints[b];pp=self.summary(s,False);aa=self.summary(a,True);bb=self.summary(b,True)
        features=list(fork_geometry(h[:4],fa[:4],fb[:4]));features.extend([np.linalg.norm((self.pos[a]+self.pos[b])/2-self.pos[s]-self.local[s])/10])
        for k in [1,2,4,7]:
            pa=fa[min(k,len(fa)-1)];pb=fb[min(k,len(fb)-1)];va=self.velocity(fa,k);vb=self.velocity(fb,k)
            features.extend([np.linalg.norm(pa-pb)/10,np.linalg.norm(va-vb)/10,np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-5)])
        ea=self.edge_one(s,a);eb=self.edge_one(s,b)
        return np.nan_to_num(np.r_[features,pp,(aa+bb)/2,np.abs(aa-bb),np.minimum(ea[:48],eb[:48]),np.maximum(ea[:48],eb[:48])]).astype(np.float32)
    def rows(self,kind,rows):
        fn=self.edge_one if kind=='edge' else self.fork_one
        return np.stack([fn(*map(int,row)) for row in rows]).astype(np.float32)
