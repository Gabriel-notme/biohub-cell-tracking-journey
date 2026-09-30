from pathlib import Path
import json,numpy as np,torch
from scipy.special import expit
from track_set_refine import TrackSetRefiner
from track_set import mask_incumbent,event_probabilities
from evaluate_track_stage import CONFIGS,TRAIN
R=Path('/workspace/biohub');torch.set_num_threads(2);split=json.loads((R/'track_data/split.json').read_text());name=split['train'][0]
raw=json.loads((R/'b4_training_graphs'/(name+'.json')).read_text());nodes={int(k):v for k,v in raw['nodes'].items()};paths=[R/n for n in ['TrackSet3_137/best.pt','TrackSet3_823/best.pt','TrackRobust_137/best.pt']]
obj=TrackSetRefiner(paths,TRAIN,CONFIGS['wide_preserve'],R/'track_prediction_cache');pred=obj.predict(name,nodes,raw['edges']);prob=expit(pred['edge_logits'])
with np.load(R/'track_corrupt3'/(name+'.npz')) as d:rows=d['edge'];ids=d['ids']
with np.load(R/'TrackSet3_features'/(name+'.npz')) as a:x=a['edge']
lookup={tuple(map(int,row)):i for i,row in enumerate(pred['edge'])};errors=[];feature_errors=[];fp32_errors=[];rank_stable=[];gate_stable=[]
for feature_file in (R/'track_prediction_cache').glob('setfeatures_'+name+'_*.npz'):
    with np.load(feature_file) as d:
        if d['features'].shape[1]==597 and np.array_equal(d['rows'],pred['edge']):full_features=d['features'];break
for source in np.unique(rows[:,0]):
    ix=np.flatnonzero(rows[:,0]==source);actual=ids[rows[ix,:-1]]
    if len(actual)!=int((pred['edge'][:,0]==actual[0,0]).sum()):continue
    selected=[lookup[tuple(map(int,row))] for row in actual];xx=torch.from_numpy(x[ix][None]).cuda();valid=torch.ones(xx.shape[:2],dtype=torch.bool,device='cuda')
    full=torch.from_numpy(full_features[selected][None]).cuda();feature_errors.append([float(np.max(np.abs(x[ix,:338]-full_features[selected,:338]))),float(np.max(np.abs(x[ix,338:]-full_features[selected,338:])))])
    for mi,model in enumerate(obj.heads):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):outputs=model(mask_incumbent(xx),valid)
        p,_=event_probabilities(*outputs,valid);reference=p[0].cpu().numpy();observed=prob[mi,selected];errors.append(float(np.max(np.abs(reference-observed))));rank_stable.append(int(reference.argmax())==int(observed.argmax()));gate_stable.extend(bool(np.array_equal(reference>=threshold,observed>=threshold)) for threshold in [.2,.8,.9,.98])
        with torch.inference_mode():
            p1,_=event_probabilities(*model(mask_incumbent(xx),valid),valid);p2,_=event_probabilities(*model(mask_incumbent(full),valid),valid);fp32_errors.append(float((p1-p2).abs().max()))
    if len(errors)>=40:break
print('FEATURE_PARITY_DIAGNOSTIC',json.dumps({'physical_feature_max':max(x[0] for x in feature_errors),'image_feature_max':max(x[1] for x in feature_errors),'fp32_probability_max':max(fp32_errors),'mixed_probability_max':max(errors)}),flush=True)
assert len(errors)>=2 and max(x[0] for x in feature_errors)==0 and max(x[1] for x in feature_errors)<=.02 and max(errors)<.005 and all(rank_stable) and all(gate_stable)
report={'movie_count':1,'groups_checked':len(errors)//2,'maximum_probability_error':max(errors),'maximum_physical_feature_error':max(x[0] for x in feature_errors),'maximum_image_feature_error':max(x[1] for x in feature_errors),'candidate_ranking_preserved':all(rank_stable),'tested_probability_gates_preserved':all(gate_stable),'numerical_note':'Half-precision image encoders use different compact-bank and full-graph batch shapes. Small rounding differences are measured rather than claimed to be byte-identical; ranking and tested decision gates must be unchanged.','training_and_full_graph_feature_parity':True}
(R/'robust_set_feature_parity_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
