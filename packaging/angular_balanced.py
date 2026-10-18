"""Fixed even/odd kernel follow-up, not a replacement of archived primary methods."""
import hashlib
import numpy as np
from angular_null import normalize,tyler_shape,whiten
ARMS=('D4','K','KE','KO','KB')
def component_scores(w):
    g=w@w.T;q=w.shape[1];e=float(np.exp(-2)*np.cosh(2*g).mean());o=float(np.exp(-2)*np.sinh(2*g).mean())
    return dict(D4=float(np.mean(g**4)-3/(q*(q+2))),K=e+o,KE=e,KO=o)
def evaluate(x,*,draws,seed):
    if draws<2:raise ValueError('at least two draws required')
    x=normalize(x);shape,fit=tyler_shape(x);observed=component_scores(whiten(x,shape))
    rng=np.random.default_rng(seed);refs={a:[] for a in observed};max_iterations=0;max_residual=0.
    for _ in range(draws):
        u=normalize(rng.standard_normal(x.shape));s,info=tyler_shape(u);r=component_scores(whiten(u,s))
        for a,v in r.items():refs[a].append(v)
        max_iterations=max(max_iterations,info['iterations']);max_residual=max(max_residual,info['whitened_moment_residual'])
    p={a:float((1+np.count_nonzero(np.asarray(v)>=observed[a]))/(draws+1)) for a,v in refs.items()}
    p['KB']=min(1.,2*min(p['KE'],p['KO']))
    return dict(p=p,observed=observed,shape_fit=fit,max_reference_iterations=max_iterations,max_reference_residual=max_residual,
        data_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest())
