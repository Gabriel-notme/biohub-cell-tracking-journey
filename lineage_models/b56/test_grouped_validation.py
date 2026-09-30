import numpy as np
from grouped_validation import grouped_mask
names=[str(i) for i in range(20)];groups=np.repeat(np.arange(20),10);y=np.zeros(200);y[0]=1;y[10]=1
for seed in [137,823,2029]:
    mask,info=grouped_mask(names,groups,y,seed)
    assert info['usable'] and set(info['fitting_movies']).isdisjoint(info['validation_movies'])
    assert y[mask].sum()==1 and y[~mask].sum()==1
    assert np.array_equal(mask,grouped_mask(names,groups,y,seed)[0])
y[10]=0;mask,info=grouped_mask(names,groups,y,137);assert not info['usable'] and y[~mask].sum()==1
print('Movie-level rare-event validation: disjoint, positive on both sides when feasible, deterministic, explicit insufficient-group fallback.')
