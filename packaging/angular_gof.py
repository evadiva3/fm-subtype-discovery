"""Fixed-kernel and published-bootstrap comparators; see docs/PROTOCOL_V9.md."""
from __future__ import annotations
import hashlib
import numpy as np
from angular_null import normalize,tyler_shape,whiten,statistic

VERSION='2026-09-15.v9.1'
ARMS=('W','D4','K','E')

def kernel_mean(x,y=None,gamma=1.):
    y=x if y is None else y
    return float(np.exp(-gamma*np.maximum(0.,2.-2.*(x@y.T))).mean())

def two_sample_kernel(x,y,gamma=1.):
    """Ebner et al. Eq. 22, xi=2; x/y have unit-length rows; diagonal pairs included."""
    n,m=len(x),len(y)
    return n*m/(n+m)*(kernel_mean(x,gamma=gamma)+kernel_mean(y,gamma=gamma)-2*kernel_mean(x,y,gamma))

def shape_factor(shape):
    eigen,vectors=np.linalg.eigh(shape)
    if np.any(eigen<=0):raise ArithmeticError('nonpositive shape')
    return np.sqrt(eigen)[:,None]*vectors.T

def whitened_scores(w):
    q=w.shape[1];sil,k=statistic(w)
    gram=w@w.T
    return {'W':sil,'D4':float(np.mean(gram**4)-3/(q*(q+2))),
            'K':float(np.exp(-np.maximum(0.,2-2*gram)).mean())},k

def evaluate(x,*,draws,seed,bootstrap_seed,auxiliary=200):
    if draws<2 or auxiliary<2:raise ValueError('at least two reference and auxiliary draws required')
    x=normalize(x);n,q=x.shape
    shape,fit=tyler_shape(x);factor=shape_factor(shape)
    observed,k=whitened_scores(whiten(x,shape))
    rng=np.random.default_rng(seed);boot=np.random.default_rng(bootstrap_seed)
    y=normalize(boot.standard_normal((auxiliary,q))@factor)
    observed['E']=two_sample_kernel(x,y)
    refs={a:np.empty(draws) for a in ARMS};max_iterations=0;max_residual=0.;guard=0
    for b in range(draws):
        uniform=normalize(rng.standard_normal((n,q)))
        sphere_fit,info=tyler_shape(uniform)
        vals,ref_k=whitened_scores(whiten(uniform,sphere_fit))
        guard+=ref_k is None
        for a,value in vals.items():refs[a][b]=value
        star=normalize(boot.standard_normal((n,q))@factor)
        star_fit,star_info=tyler_shape(star)
        ystar=normalize(boot.standard_normal((auxiliary,q))@shape_factor(star_fit))
        refs['E'][b]=two_sample_kernel(star,ystar)
        max_iterations=max(max_iterations,info['iterations'],star_info['iterations'])
        max_residual=max(max_residual,info['whitened_moment_residual'],star_info['whitened_moment_residual'])
    p={a:float((1+np.count_nonzero(refs[a]>=observed[a]))/(draws+1)) for a in ARMS}
    return dict(p=p,observed=observed,selected_k=k,guard_failed=k is None,
        reference_guard_failures=guard,shape_fit=fit,max_reference_iterations=max_iterations,
        max_reference_residual=max_residual,null_mean={a:float(v.mean()) for a,v in refs.items()},
        data_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest())
