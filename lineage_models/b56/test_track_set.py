from pathlib import Path
import json,torch
from track_set import TrackSetNet,mask_incumbent,event_probabilities
torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(7)
model=TrackSetNet(input_dim=856).cuda().eval();x=torch.randn(3,6,856,device='cuda');valid=torch.arange(6,device='cuda')[None,:]<torch.tensor([6,4,1],device='cuda')[:,None];permutation=torch.tensor([3,0,5,2,1,4],device='cuda');inverse=permutation.argsort()
with torch.inference_mode():
    edge,count,pair=model(mask_incumbent(x),valid);p,fp=event_probabilities(edge,count,pair,valid)
    ee,cc,ff=model(mask_incumbent(x[:,permutation]),valid[:,permutation]);pp,pf=event_probabilities(ee,cc,ff,valid[:,permutation]);pf=pf+pf.transpose(1,2);fp_symmetric=fp+fp.transpose(1,2)
    edge_error=float((p-pp[:,inverse]).abs().max());pair_error=float((fp_symmetric-pf[:,inverse][:,:,inverse]).abs().max());count_error=float((count-cc).abs().max())
    padded=x.clone();padded[~valid]=10000.;eee,ccc,fff=model(mask_incumbent(padded),valid);padded_error=float((edge[valid]-eee[valid]).abs().max())
    assert max(edge_error,pair_error,count_error,padded_error)<2e-5
    assert torch.allclose(fp.sum((1,2))[:2],count.softmax(-1)[:2,2],atol=1e-6) and float(fp[2].sum())==0.
    sp,sf=event_probabilities(edge,count,pair,valid,structured=True)
    spp,sff=event_probabilities(ee,cc,ff,valid[:,permutation],structured=True)
    assert torch.allclose(sp,spp[:,inverse],atol=2e-5)
    expected=count.softmax(-1)[:,1]+2*sf.sum((1,2))
    assert torch.allclose(sp.sum(1),expected,atol=1e-6)
    assert bool(((sp>=0)&(sp<=1)).all())
report={'candidate_permutation_equivariant':True,'daughter_pair_symmetric':True,'padding_cannot_change_valid_predictions':True,'division_pair_mass_equals_division_probability':True,'structured_edge_marginal_mass_equals_expected_child_count':True,'maximum_error':max(edge_error,pair_error,count_error,padded_error)}
Path('/workspace/biohub/track_set_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
