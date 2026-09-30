"""Vectorized implementation of the existing, unchanged motion descriptors."""
import numpy as np
from motion_features import MotionFeatures

def dot(a,b):return np.vecdot(a,b) if hasattr(np,'vecdot') else np.einsum('...i,...i->...',a,b)
def norm(x):return np.sqrt(dot(x,x))
def div(x,n):return x/np.asarray(n,dtype=x.dtype)[:,None]

class FastMotionFeatures(MotionFeatures):
    def arrays(self):
        if hasattr(self,'_fast'):return self._fast
        ids=list(self.nodes);lookup={n:i for i,n in enumerate(ids)};p=np.stack([self.pos[n] for n in ids]);photo=np.stack([self.photo[n] for n in ids]);density=np.stack([self.density[n] for n in ids]);spread=np.asarray([self.flow_spread[n] for n in ids],np.float32);result={'lookup':lookup,'p':p,'density':density,'local':np.stack([self.local[n] for n in ids]),'outdegree':np.asarray([len(self.out.get(n,[])) for n in ids]),'incoming':np.asarray([int(n in self.prev) for n in ids])}
        for key,tracks in [('h',self.hist),('f',self.future)]:
            lengths=np.asarray([len(tracks[n]) for n in ids]);indices=np.asarray([[lookup[k] for k in tracks[n]]+[lookup[tracks[n][-1]]]*(8-len(tracks[n])) for n in ids]);points=p[indices];photos=photo[indices];mask=np.arange(8)[None,:]<lengths[:,None];m=mask[:,:,None]
            mean=div((photos*m).sum(1),lengths);sd=np.sqrt(div((((photos-mean[:,None])**2)*m).sum(1),lengths));n3=np.minimum(lengths,3);mask3=np.arange(8)[None,:]<n3[:,None];first3=div((photos*mask3[:,:,None]).sum(1),n3)
            delta=np.diff(points,axis=1);speed=norm(delta);acc=norm(np.diff(delta,axis=1));ns=np.maximum(lengths-1,1);na=np.maximum(lengths-2,1);sm=np.arange(7)[None,:]<np.maximum(lengths-1,0)[:,None];am=np.arange(6)[None,:]<np.maximum(lengths-2,0)[:,None]
            speed_mean=(speed*sm).sum(1)/ns.astype(speed.dtype);speed_std=np.sqrt(((speed-speed_mean[:,None])**2*sm).sum(1)/ns.astype(speed.dtype));acc_mean=(acc*am).sum(1)/na.astype(acc.dtype)
            summary=np.column_stack([photos[:,0],first3,mean,sd,photos[np.arange(len(ids)),lengths-1]-photos[:,0],lengths/8,speed_mean/10,(speed*sm).max(1)/10,speed_std/10,acc_mean/10,(acc*am).max(1)/10,density/20,spread/10]).astype(np.float32)
            result[key]=points;result[key+'len']=lengths;result[key+'summary']=summary
        self._fast=result;return result
    def edge_rows(self,rows,head_only=False):
        d=self.arrays();lookup=d['lookup'];ix=np.asarray([[lookup[int(s)],lookup[int(t)]] for s,t in rows]);s,t=ix.T;p=d['p'][s];q=d['p'][t];delta=q-p;h=d['h'][s];f=d['f'][t];hl=d['hlen'][s];fl=d['flen'][t];ii=np.arange(len(rows));kh=np.minimum(hl-1,3);kf=np.minimum(fl-1,3);v=div(p-h[ii,kh],np.maximum(kh,1));w=div(f[ii,kf]-q,np.maximum(kf,1))
        parts=[delta/10,norm(delta)/10,v/10,(delta-v)/10,w/10,norm(w-v)/10,hl>1,fl>1,norm(delta)/10,norm(delta-d['local'][s])/10,norm(delta-d['local'][t])/10,norm(d['local'][s]-d['local'][t])/10]
        for k in [1,2,4,7]:
            kh=np.minimum(k,hl-1);kf=np.minimum(k,fl-1);vv=div(p-h[ii,kh],np.maximum(kh,1));ww=div(f[ii,kf]-q,np.maximum(kf,1))
            parts.extend([norm(delta-vv)/10,norm(delta-ww)/10,norm(vv-ww)/10,dot(vv,ww)/(norm(vv)*norm(ww)+1e-5),norm(vv)/10,norm(ww)/10])
        original=[self.original.get((int(a),int(b)),{}) for a,b in rows];parts.extend([[e.get('edge_prob') if e.get('edge_prob') is not None else .5 for e in original],[(int(a),int(b)) in self.original for a,b in rows],d['outdegree'][s],d['incoming'][t]])
        if not head_only:
            a=d['hsummary'][s];b=d['fsummary'][t];parts.extend([a,b,np.abs(a[:,:24]-b[:,:24]),d['density'][t]-d['density'][s]])
        return np.nan_to_num(np.column_stack(parts)).astype(np.float32)
    def fork_rows(self,rows):
        d=self.arrays();lookup=d['lookup'];ix=np.asarray([[lookup[int(n)] for n in row] for row in rows]);s,a,b=ix.T;ii=np.arange(len(rows));p=d['p'][s];qa=d['p'][a];qb=d['p'][b];h=d['h'][s];fa=d['f'][a];fb=d['f'][b];hl=d['hlen'][s];al=d['flen'][a];bl=d['flen'][b];kh=np.minimum(hl-1,3);v=div(p-h[ii,kh],np.maximum(kh,1));da=qa-p;db=qb-p;na=norm(da);nb=norm(db)
        parts=[np.minimum(na,nb)/10,np.maximum(na,nb)/10,norm(qa-qb)/10,dot(da,db)/(na*nb+1e-4),norm((qa+qb)/2-p-v)/10,np.abs(na-nb)/10,norm(v)/10]
        for k in range(4):
            pa=fa[ii,np.minimum(k,al-1)];pb=fb[ii,np.minimum(k,bl-1)];xa=norm(pa-p);xb=norm(pb-p)
            parts.extend([norm(pa-pb)/10,norm((pa+pb)/2-p-(k+1)*v)/10,np.minimum(xa,xb)/10,np.maximum(xa,xb)/10])
        ka=np.minimum(3,al-1);kb=np.minimum(3,bl-1);va=div(fa[ii,ka]-qa,np.maximum(ka,1));vb=div(fb[ii,kb]-qb,np.maximum(kb,1))
        parts.extend([norm(va-vb)/10,dot(va,vb)/(norm(va)*norm(vb)+1e-4),np.minimum(hl,4)/4,np.minimum(np.minimum(al,bl),4)/4,np.minimum(np.maximum(al,bl),4)/4,norm((qa+qb)/2-p-d['local'][s])/10])
        for k in [1,2,4,7]:
            ka=np.minimum(k,al-1);kb=np.minimum(k,bl-1);pa=fa[ii,ka];pb=fb[ii,kb];va=div(pa-qa,np.maximum(ka,1));vb=div(pb-qb,np.maximum(kb,1))
            parts.extend([norm(pa-pb)/10,norm(va-vb)/10,dot(va,vb)/(norm(va)*norm(vb)+1e-5)])
        ea=self.edge_rows(rows[:,[0,1]],True);eb=self.edge_rows(rows[:,[0,2]],True);aa=d['fsummary'][a];bb=d['fsummary'][b];parts.extend([d['hsummary'][s],(aa+bb)/2,np.abs(aa-bb),np.minimum(ea,eb),np.maximum(ea,eb)])
        return np.nan_to_num(np.column_stack(parts)).astype(np.float32)
    def rows(self,kind,rows):
        rows=np.asarray(rows,np.int64)
        if not len(rows):return np.empty((0,338 if kind=='edge' else 530),np.float32)
        return self.edge_rows(rows) if kind=='edge' else self.fork_rows(rows)
