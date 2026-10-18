"""Shape-aware ACG diagnostics. Fixed projection and independent rows are required."""
from __future__ import annotations
import hashlib
import numpy as np
from nullcal_validation import normalize,scores,stream_seed

VERSION='2026-09-15.1'

def fixed_projection(d,q,seed):
    if not 2<=q<=d:raise ValueError('require 2 <= q <= d')
    a=np.random.default_rng(seed).standard_normal((d,q))
    return np.linalg.qr(a,mode='reduced')[0]

def tyler_shape(x,*,tol=1e-9,max_iter=1000):
    x=normalize(x);n,q=x.shape
    if n<=q:raise ValueError('unregularized Tyler shape requires n > dimension')
    s=np.linalg.svd(x,compute_uv=False)
    if s[-1]<=1e-12*s[0]:raise ValueError('directions are numerically rank deficient')
    shape=np.eye(q)
    for iteration in range(1,max_iter+1):
        invx=np.linalg.solve(shape,x.T).T
        distances=np.sum(invx*x,axis=1)
        if not np.isfinite(distances).all() or np.any(distances<=0):
            raise ArithmeticError('invalid Tyler quadratic form')
        update=(x.T/distances)@x*(q/n)
        update*=q/np.trace(update);update=(update+update.T)/2
        relative=np.linalg.norm(update-shape,'fro')/np.linalg.norm(shape,'fro')
        shape=update
        if relative<tol:
            eig,vec=np.linalg.eigh(shape)
            if eig[0]<=0:raise ArithmeticError('Tyler shape not positive definite')
            w=normalize(x@(vec/np.sqrt(eig)))
            residual=np.linalg.norm((q/n)*(w.T@w)-np.eye(q),'fro')
            if residual<1e-7:
                return shape,dict(iterations=iteration,relative_change=float(relative),condition=float(eig[-1]/eig[0]),whitened_moment_residual=float(residual))
    raise ArithmeticError('Tyler fixed-point iteration did not converge')

def whiten(x,shape):
    eig,vec=np.linalg.eigh(shape)
    return normalize(x@(vec/np.sqrt(eig))@vec.T)

def statistic(x,n_init=5):
    _,guarded,k=scores(x,n_init=n_init,cluster_seed=0)
    return max(guarded.values(),default=-1.),k

def whitened_statistic(x,n_init=5):
    shape,info=tyler_shape(x)
    value,k=statistic(whiten(x,shape),n_init)
    return value,k,info

def evaluate(x,*,draws,seed,n_init=5):
    if draws<2:raise ValueError('at least two draws required')
    x=normalize(x);n,q=x.shape
    shape,fit=tyler_shape(x)
    white=whiten(x,shape)
    raw,k_raw=statistic(x,n_init);w,k_w=statistic(white,n_init)
    mean=x.mean(axis=0)
    _,singular,vt=np.linalg.svd(x-mean,full_matrices=False)
    fitted=singular[:,None]/np.sqrt(n-1)*vt
    eig,vec=np.linalg.eigh(shape);factor=np.sqrt(eig)[:,None]*vec.T
    rng=np.random.default_rng(seed)
    ref={a:np.empty(draws) for a in ('C','T','W','N')}
    fit_iterations=[]
    for i in range(draws):
        normal=rng.standard_normal((n,q))
        ref['C'][i]=statistic(normalize(mean+normal@fitted),n_init)[0]
        ref['T'][i]=statistic(normalize(normal@factor),n_init)[0]
        sphere=normalize(normal)
        ref['N'][i]=statistic(sphere,n_init)[0]
        ref['W'][i],_,info=whitened_statistic(sphere,n_init)
        fit_iterations.append(info['iterations'])
    observed={'C':raw,'T':raw,'W':w,'N':w}
    p={a:float((1+np.count_nonzero(vals>=observed[a]))/(draws+1)) for a,vals in ref.items()}
    return dict(p=p,observed=observed,selected_k_raw=k_raw,selected_k_whitened=k_w,
                raw_guard_failed=k_raw is None,whitened_guard_failed=k_w is None,
                mean_resultant=float(np.linalg.norm(mean)),shape_fit=fit,
                max_reference_fit_iterations=max(fit_iterations),
                null_mean={a:float(v.mean()) for a,v in ref.items()},
                null_sd={a:float(v.std(ddof=1)) for a,v in ref.items()},
                null_guard_failures={a:int((v==-1).sum()) for a,v in ref.items()},
                data_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest())
