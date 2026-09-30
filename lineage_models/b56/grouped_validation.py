"""Deterministic movie-level validation that handles rare positive events."""
import hashlib
import numpy as np

def grouped_mask(names,groups,y,seed):
    counts=np.bincount(groups,weights=y,minlength=len(names));held=np.zeros(len(names),bool)
    for positive in [True,False]:
        candidates=np.flatnonzero((counts>0)==positive).tolist()
        candidates.sort(key=lambda i:hashlib.sha256((names[i]+str(seed)).encode()).hexdigest())
        n=min(max(1,round(.2*len(candidates))),len(candidates)-1) if len(candidates)>=2 else 0
        held[candidates[:n]]=True
    mask=held[groups]
    usable=bool(mask.any() and (~mask).any() and len(np.unique(y[mask]))==2 and len(np.unique(y[~mask]))==2)
    return mask,{'usable':usable,'fitting_movies':[n for n,h in zip(names,held) if not h],'validation_movies':[n for n,h in zip(names,held) if h],'fitting_positives':int(y[~mask].sum()),'validation_positives':int(y[mask].sum())}
