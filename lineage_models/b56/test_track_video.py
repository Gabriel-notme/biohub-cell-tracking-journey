"""Check that vectorized inference geometry equals the training definitions."""
import numpy as np,torch,json
from track_video import TrackVideoNet,sequences,PATCH,STEPS
from track_video_refine import geometries
from refine_events import structure
from cell_event import chain,edge_geometry,fork_geometry
torch.set_num_threads(2);rng=np.random.default_rng(124)
nodes={};edges=[]
for c in range(12):
    p=rng.uniform(10,40,3)
    for t in range(12):
        n=c*100+t;p=p+rng.normal(0,.6,3);nodes[n]={'t':t,'z':float(p[0]),'y':float(p[1]),'x':float(p[2])}
        if t:edges.append({'source_id':n-1,'target_id':n})
out,prev,frames,pos=structure(nodes,edges);ids=sorted(nodes);ix={n:i for i,n in enumerate(ids)};seq,m=sequences(ids,nodes,out,prev);positions=np.asarray([pos[n] for n in ids])
maxerr={}
for kind in ['edge','fork']:
    rows=[];expected=[]
    for t in range(11):
        for s in frames[t]:
            a,b=rng.choice(frames[t+1],2,replace=False)
            rr=[s,int(a)] if kind=='edge' else [s,int(a),int(b)];rows.append([ix[n] for n in rr])
            expected.append(edge_geometry(chain(s,prev,pos),chain(a,out,pos)) if kind=='edge' else fork_geometry(chain(s,prev,pos),chain(a,out,pos),chain(b,out,pos)))
    got=geometries(np.asarray(rows),kind,positions,seq,m);maxerr[kind]=float(np.abs(got-np.stack(expected)).max());assert np.allclose(got,expected,atol=2e-6),maxerr
device='cuda' if torch.cuda.is_available() else 'cpu';model=TrackVideoNet().to(device).eval();x=torch.rand(2,3,STEPS,*PATCH,device=device);motion=torch.rand(2,3,STEPS,4,device=device);motion[...,3]=1;g=torch.rand(2,28,device=device)
with torch.no_grad():
    full,_=model(x,motion,g,'fork');z=model.encode_frame(x.reshape(-1,1,*PATCH)).reshape(6,STEPS,64);z=model.encode_track(z,motion.reshape(6,STEPS,4)).reshape(2,3,64);modular=model.classify(z,g,'fork');assert torch.equal(full,modular)
    swapped=model.classify(z[:,[0,2,1]],g,'fork');assert torch.equal(full,swapped),'Daughter order must not matter'
print(json.dumps({'geometry_max_error':maxerr,'train_infer_identical':True,'daughter_swap_identical':True}))
