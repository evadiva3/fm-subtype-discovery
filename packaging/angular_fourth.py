"""Fourth-moment follow-up: see docs/PROTOCOL_V8_FOLLOWUP.md."""
import hashlib
import numpy as np
from angular_null import normalize,tyler_shape,whiten

def discrepancy(w):
    w=normalize(w);q=w.shape[1]
    return float(np.mean((w@w.T)**4)-3/(q*(q+2)))

def fitted_discrepancy(x):
    shape,info=tyler_shape(x)
    return discrepancy(whiten(x,shape)),info

def evaluate(x,*,draws,seed):
    if draws<2:raise ValueError('at least two draws required')
    x=normalize(x);value,fit=fitted_discrepancy(x)
    rng=np.random.default_rng(seed);ref=[];iterations=[]
    for _ in range(draws):
        v,info=fitted_discrepancy(rng.standard_normal(x.shape))
        ref.append(v);iterations.append(info['iterations'])
    return dict(p=float((1+np.count_nonzero(np.asarray(ref)>=value))/(draws+1)),
        statistic=value,shape_fit=fit,max_reference_fit_iterations=max(iterations),
        null_mean=float(np.mean(ref)),null_sd=float(np.std(ref,ddof=1)),
        data_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest())
