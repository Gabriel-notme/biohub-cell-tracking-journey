"""Two-frame missing-cell bridges, independently verified at every new link."""
from pathlib import Path
import os,json,hashlib
import numpy as np,torch
from verified_recovery import VerifiedRecoveryRefiner
from cell_event import Movie,SCALE,chain,edge_geometry
from refine_events import structure
class MultiFrameRecoveryRefiner(VerifiedRecoveryRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        super().__init__(paths,root,{'bridge_gap':2,**(config or {})},cache_dir)
        assert int(self.config['bridge_gap'])>1
    @torch.inference_mode()
    def proposals(self,name,nodes,edges):
        proposals=super().proposals(name,nodes,edges)
        if not proposals:return []
        cfg={k:v for k,v in self.config.items() if k!='bridge_verification_threshold'};identity=hashlib.sha256(json.dumps([nodes,edges,cfg],sort_keys=True).encode()).hexdigest()[:16];path=self.cache/('bridge_confirm_'+name+'_'+self.signature+'_'+identity+'.json')
        if path.exists():records=json.loads(path.read_text())
        else:
            out,prev,_,pos=structure(nodes,edges);gap=int(self.config['bridge_gap']);requests=[];lookup={};rows=[];geom=[];owners=[]
            def request(t,q):
                key=(int(t),*np.asarray(q).tolist())
                if key not in lookup:lookup[key]=len(requests);requests.append((t,q/SCALE))
                return lookup[key]
            for pi,r in enumerate(proposals):
                s,d=r['s'],r['d'];t=int(nodes[s]['t']);q=np.asarray(r['position']);points=[q+(pos[d]-q)*j/gap for j in range(gap)]+[pos[d]]
                for j in range(gap):
                    rows.append([request(t+1+j,points[j]),request(t+2+j,points[j+1])]);history=np.vstack(list(reversed(points[:j+1]))+list(chain(s,prev,pos)))[:4];future=np.vstack(points[j+1:]+list(chain(d,out,pos)[1:]))[:4];geom.append(edge_geometry(history,future));owners.append(pi)
            bank=torch.empty((len(requests),3,64),device='cuda');movie=Movie(self.root/(name+'.zarr'),context=7)
            for start in range(0,len(requests),192):
                req=requests[start:start+192];x=torch.from_numpy(movie.patches([t for t,q in req],[q for t,q in req])).cuda().float()
                with torch.autocast('cuda',dtype=torch.float16):z=self.verifier.base.encode(torch.cat([x[:,:3],x[:,2:5],x[:,4:7]],0))
                bank[start:start+len(req)]=z.reshape(3,len(req),64).transpose(0,1).float()
            minima=np.ones(len(proposals))
            for start in range(0,len(rows),512):
                ix=torch.tensor(rows[start:start+512],device='cuda');gg=torch.from_numpy(np.stack(geom[start:start+512])).cuda()
                with torch.autocast('cuda',dtype=torch.float16):p=self.verifier(bank[ix],gg,'edge')
                for owner,prob in zip(owners[start:start+512],p.float().sigmoid().cpu().tolist()):minima[owner]=min(minima[owner],prob)
            records=[{**r,'bridge_probability_min':float(p)} for r,p in zip(proposals,minima)];tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(records));os.replace(tmp,path)
        return [r for r in records if r['bridge_probability_min']>=self.config.get('bridge_verification_threshold',.99)]
