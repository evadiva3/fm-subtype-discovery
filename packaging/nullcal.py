from __future__ import annotations
from dataclasses import dataclass,field
from typing import Iterable,Sequence
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
__all__=[
    "NullResult",
    "LadderResult",
    "CalibrationResult",
    "l2_normalize",
    "participation_ratio",
    "separation_ratio",
    "best_silhouette",
    "null_test",
    "ladder",
    "calibrate_paired",
    "type1_surface_cell",
    "propagated_null_checklist",
    ]
DEFAULT_K_GRID:tuple[int,...]=(2,3,4,5,6)
_DEGENERATE=-1.0

def l2_normalize(X:np.ndarray,eps:float=1e-12)->np.ndarray:
    X=np.asarray(X,dtype=np.float64)
    no=np.linalg.norm(X,axis=1,keepdims=True)
    return X/np.maximum(no,eps)


def participation_ratio(X:np.ndarray,normalize:bool=False)->float:
    X=np.asarray(X,dtype=np.float64)
    if normalize:
        X=l2_normalize(X)
    Xc=X-X.mean(axis=0,keepdims=True)
    n=Xc.shape[0]
    if n<2:
        raise ValueError("need at least two rows")
    lam=np.linalg.svd(Xc,compute_uv=False)**2/(n-1)
    lam=lam[lam>0]
    if lam.size==0:
        return 0.0
    return float(lam.sum()**2/(lam**2).sum())


def min_cluster_guard(n:int)->int:
    return max(4,int(np.rint(0.15*n)))


def separation_ratio(X:np.ndarray,labels:np.ndarray)->float:
    Xn=l2_normalize(X)
    labels=np.asarray(labels)
    un=np.unique(labels)
    if un.size<2:
        raise ValueError("need at least two groups")
    c=np.stack([Xn[labels==u].mean(axis=0)for u in un])
    d=[]
    for i in range(len(un)):
        for j in range(i+1,len(un)):
            d.append(np.linalg.norm(c[i]-c[j]))
    b=float(np.mean(d))
    sq=np.concatenate(
        [
        np.sum((Xn[labels==u]-c[i])**2,axis=1)
        for i,u in enumerate(un)
        ]
        )
    wr=float(np.sqrt(sq.mean()))
    if wr==0.0:
        return float("inf")
    return b/wr


def best_silhouette(
    X:np.ndarray,
    k_grid:Sequence[int]=DEFAULT_K_GRID,
    min_size:int|None=None,
    n_init:int=20,
    random_state:int|None=0,
    )->tuple[float,int|None,np.ndarray|None]:
    X=np.asarray(X,dtype=np.float64)
    n=X.shape[0]
    if min_size is None:
        min_size=min_cluster_guard(n)
    b=(_DEGENERATE,None,None)
    for k in k_grid:
        if k<2 or k>=n:
            continue
        km=KMeans(n_clusters=k,n_init=n_init,random_state=random_state)
        l=km.fit_predict(X)
        c=np.bincount(l,minlength=k)
        if c.min()<min_size:
            continue
        if len(np.unique(l))<2:
            continue
        s=float(silhouette_score(X,l))
        if s>b[0]:
            b=(s,int(k),l)
    return b


def silhouette_at_k(
    X:np.ndarray,
    k:int,
    n_init:int=20,
    random_state:int|None=0,
    )->float:
    X=np.asarray(X,dtype=np.float64)
    km=KMeans(n_clusters=k,n_init=n_init,random_state=random_state)
    l=km.fit_predict(X)
    if len(np.unique(l))<2:
        return _DEGENERATE
    return float(silhouette_score(X,l))


def _gaussian_factor(X:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    X=np.asarray(X,dtype=np.float64)
    n=X.shape[0]
    m=X.mean(axis=0)
    Xc=X-m
    _,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    A=(S[:,None]/np.sqrt(n-1))*Vt
    return m,A


def _draw(mean:np.ndarray,A:np.ndarray,n:int,rng:np.random.Generator)->np.ndarray:
    z=rng.standard_normal((n,A.shape[0]))
    return mean+z@A


@dataclass


class NullResult:
    p:float
    observed:float
    observed_k:int|None
    null_mean:float
    null_sd:float
    n_draws:int
    n_exceed:int
    reproject:bool
    match_selection:bool
    null_k_counts:dict[int,int]=field(default_factory=dict)
    n_degenerate:int=0
    @property
    def construction(self)->str:
        return{
            (False,False):"M (misspecified)",
            (True,False):"G (geometry corrected)",
            (False,True):"S (selection corrected)",
            (True,True):"C (fully corrected)",
            }[(self.reproject,self.match_selection)]
    def __repr__(self)->str:
        return(
            f"<{self.construction}: p={self.p:.4f} "
            f"observed={self.observed:.4f} at k={self.observed_k} "
            f"null_mean={self.null_mean:.4f} ({self.n_draws} draws)>"
            )


def null_test(
    X:np.ndarray,
    *,
    reproject:bool=True,
    match_selection:bool=True,
    k_grid:Sequence[int]=DEFAULT_K_GRID,
    n_draws:int=20000,
    min_size:int|None=None,
    n_init:int=20,
    seed:int=0,
    observed:tuple[float,int]|None=None,
    )->NullResult:
    X=np.asarray(X,dtype=np.float64)
    n=X.shape[0]
    if min_size is None:
        min_size=min_cluster_guard(n)
    Xn=l2_normalize(X)
    if observed is None:
        os,ok,_=best_silhouette(
            Xn,k_grid=k_grid,min_size=min_size,n_init=n_init,random_state=seed
            )
    else:
        os,ok=observed
    if ok is None:
        raise ValueError(
            "no value of k passes the minimum-cluster-size guard on the "
            "observed data; there is no statistic to test"
            )
    m,A=_gaussian_factor(Xn)
    rng=np.random.default_rng(seed)
    st=np.empty(n_draws,dtype=np.float64)
    kc:dict[int,int]={int(k):0 for k in k_grid}
    nd=0
    for b in range(n_draws):
        D=_draw(m,A,n,rng)
        if reproject:
            D=l2_normalize(D)
        if match_selection:
            s,ks,_=best_silhouette(
                D,k_grid=k_grid,min_size=min_size,n_init=n_init,random_state=b
                )
            if ks is None:
                nd+=1
            else:
                kc[ks]+=1
        else:
            s=silhouette_at_k(D,ok,n_init=n_init,random_state=b)
            kc[int(ok)]+=1
        st[b]=s
    ne=int(np.sum(st>=os))
    p=(ne+1)/(n_draws+1)
    return NullResult(
        p=float(p),
        observed=float(os),
        observed_k=int(ok),
        null_mean=float(st.mean()),
        null_sd=float(st.std(ddof=1)),
        n_draws=n_draws,
        n_exceed=ne,
        reproject=reproject,
        match_selection=match_selection,
        null_k_counts=kc,
        n_degenerate=nd,
        )


@dataclass


class LadderResult:
    M:NullResult
    G:NullResult
    S:NullResult
    C:NullResult
    @property
    def null_mean_spread(self)->float:
        m=[r.null_mean for r in(self.M,self.G,self.S,self.C)]
        return float(max(m)-min(m))
    @property
    def p_spread(self)->float:
        ps=[r.p for r in(self.M,self.G,self.S,self.C)]
        return float(max(ps)-min(ps))
    def table(self)->str:
        ro=[
            ("M  misspecified",self.M),
            ("G  geometry corrected",self.G),
            ("S  selection corrected",self.S),
            ("C  fully corrected",self.C),
            ]
        out=[
            f"observed silhouette {self.C.observed:.4f} at k={self.C.observed_k}, "
            f"{self.C.n_draws} draws",
            "",
            f"{'construction':<26}{'p':>10}{'null mean':>12}{'null sd':>10}",
            "-"*58,
            ]
        for na,r in ro:
            out.append(f"{na:<26}{r.p:>10.4f}{r.null_mean:>12.4f}{r.null_sd:>10.4f}")
        out+=[
            "-"*58,
            f"{'spread':<26}{self.p_spread:>10.4f}{self.null_mean_spread:>12.4f}",
            "",
            "null's own selected k under C: "
            +", ".join(f"{k}:{v}" for k,v in sorted(self.C.null_k_counts.items())),
            ]
        if self.C.null_k_counts:
            t=sum(self.C.null_k_counts.values())
            ao=self.C.null_k_counts.get(int(self.C.observed_k),0)
            if t:
                out.append(
                    f"  {100 * (1 - ao / t):.1f}% of draws maximize at a k "
                    f"different from the observed selection of {self.C.observed_k}"
                    )
        return "\n".join(out)


def ladder(
    X:np.ndarray,
    *,
    k_grid:Sequence[int]=DEFAULT_K_GRID,
    n_draws:int=20000,
    min_size:int|None=None,
    n_init:int=20,
    seed:int=0,
    )->LadderResult:
    Xn=l2_normalize(X)
    obs=best_silhouette(
        Xn,k_grid=k_grid,min_size=min_size,n_init=n_init,random_state=seed
        )
    o=(obs[0],obs[1])
    kw=dict(
        k_grid=k_grid,
        n_draws=n_draws,
        min_size=min_size,
        n_init=n_init,
        seed=seed,
        observed=o,
        )
    return LadderResult(
        M=null_test(X,reproject=False,match_selection=False,**kw),
        G=null_test(X,reproject=True,match_selection=False,**kw),
        S=null_test(X,reproject=False,match_selection=True,**kw),
        C=null_test(X,reproject=True,match_selection=True,**kw),
        )


@dataclass


class CalibrationResult:
    label_a:str
    label_b:str
    p_a:np.ndarray
    p_b:np.ndarray
    alpha:float
    def _rate(self,p:np.ndarray)->tuple[float,tuple[float,float]]:
        n=p.size
        c=int(np.sum(p<self.alpha))
        return c/n,_wilson(c,n)
    @property
    def summary(self)->str:
        ra,cia=self._rate(self.p_a)
        rb,cib=self._rate(self.p_b)
        ao=int(np.sum((self.p_a<self.alpha)&(self.p_b>=self.alpha)))
        bo=int(np.sum((self.p_b<self.alpha)&(self.p_a>=self.alpha)))
        return "\n".join(
            [
            f"{self.p_a.size} datasets, nominal alpha = {self.alpha}",
            "",
            f"{self.label_a:<26} rejection {ra:.4f}  "
            f"[{cia[0]:.4f}, {cia[1]:.4f}]  mean p {self.p_a.mean():.4f}  "
            f"KS D {_ks_uniform(self.p_a):.4f}",
            f"{self.label_b:<26} rejection {rb:.4f}  "
            f"[{cib[0]:.4f}, {cib[1]:.4f}]  mean p {self.p_b.mean():.4f}  "
            f"KS D {_ks_uniform(self.p_b):.4f}",
            "",
            f"discordant: {ao} by '{self.label_a}' alone, "
            f"{bo} by '{self.label_b}' alone",
            "",
            "NOTE: a rejection rate resting on a handful of events is loose to "
            "roughly a factor of two under re-execution. Read it through its "
            "interval, not its point estimate. Rates resting on hundreds are tight.",
            ]
            )


def _wilson(c:int,n:int,z:float=1.959963985)->tuple[float,float]:
    if n==0:
        return(0.0,1.0)
    p=c/n
    de=1+z**2/n
    ce=(p+z**2/(2*n))/de
    h=z*np.sqrt(p*(1-p)/n+z**2/(4*n**2))/de
    return(max(0.0,ce-h),min(1.0,ce+h))


def _ks_uniform(p:np.ndarray)->float:
    p=np.sort(np.asarray(p,dtype=np.float64))
    n=p.size
    i=np.arange(1,n+1)
    return float(max(np.max(i/n-p),np.max(p-(i-1)/n)))


def calibrate_paired(
    X:np.ndarray,
    *,
    n_datasets:int=1000,
    n_draws:int=1000,
    k_grid:Sequence[int]=DEFAULT_K_GRID,
    alpha:float=0.05,
    construction_a:dict|None=None,
    construction_b:dict|None=None,
    label_a:str="M (misspecified)",
    label_b:str="C (fully corrected)",
    min_size:int|None=None,
    n_init:int=10,
    seed:int=0,
    verbose:bool=True,
    )->CalibrationResult:
    construction_a=construction_a or dict(reproject=False,match_selection=False)
    construction_b=construction_b or dict(reproject=True,match_selection=True)
    Xn=l2_normalize(np.asarray(X,dtype=np.float64))
    n,d=Xn.shape
    if min_size is None:
        min_size=min_cluster_guard(n)
    m,A=_gaussian_factor(Xn)
    rng=np.random.default_rng(seed)
    p_a=np.empty(n_datasets)
    p_b=np.empty(n_datasets)
    for i in range(n_datasets):
        D=l2_normalize(_draw(m,A,n,rng))
        obs=best_silhouette(
            D,k_grid=k_grid,min_size=min_size,n_init=n_init,random_state=i
            )
        if obs[1]is None:
            p_a[i]=p_b[i]=1.0
            continue
        kw=dict(
            k_grid=k_grid,
            n_draws=n_draws,
            min_size=min_size,
            n_init=n_init,
            seed=seed+i+1,
            observed=(obs[0],obs[1]),
            )
        p_a[i]=null_test(D,**construction_a,**kw).p
        p_b[i]=null_test(D,**construction_b,**kw).p
        if verbose and(i+1)%max(1,n_datasets//20)==0:
            print(f"  {i + 1}/{n_datasets}",flush=True)
    return CalibrationResult(label_a,label_b,p_a,p_b,alpha)


def type1_surface_cell(
    n:int,
    d:int,
    tau:float,
    *,
    n_datasets:int=120,
    n_draws:int=200,
    k_grid:Sequence[int]=DEFAULT_K_GRID,
    alpha:float=0.05,
    n_init:int=5,
    seed:int=0,
    )->dict:
    rng=np.random.default_rng(seed)
    if np.isinf(tau):
        lam=np.ones(d)
    else:
        lam=np.exp(-np.arange(d)/tau)
    Q,_=np.linalg.qr(rng.standard_normal((d,d)))
    A=np.sqrt(lam)[:,None]*Q.T
    pM,pC,prs=[],[],[]
    for i in range(n_datasets):
        r=np.random.default_rng(seed+1000*i+1)
        X=l2_normalize(r.standard_normal((n,d))@A)
        prs.append(participation_ratio(X))
        obs=best_silhouette(X,k_grid=k_grid,n_init=n_init,random_state=i)
        if obs[1]is None:
            continue
        kw=dict(k_grid=k_grid,n_draws=n_draws,n_init=n_init,
            seed=seed+1000*i+2,observed=(obs[0],obs[1]))
        pM.append(null_test(X,reproject=False,match_selection=False,**kw).p)
        pC.append(null_test(X,reproject=True,match_selection=True,**kw).p)
    pM,pC=np.array(pM),np.array(pC)
    return dict(
        n=n,d=d,tau=None if np.isinf(tau)else tau,
        pr_mean=float(np.mean(prs)),
        rej_M=float(np.mean(pM<alpha)),
        rej_C=float(np.mean(pC<alpha)),
        ci_M=_wilson(int(np.sum(pM<alpha)),pM.size),
        ci_C=_wilson(int(np.sum(pC<alpha)),pC.size),
        M_only=int(np.sum((pM<alpha)&(pC>=alpha))),
        C_only=int(np.sum((pC<alpha)&(pM>=alpha))),
        n_datasets=int(pM.size),
        )


def propagated_null_checklist()->str:
    return "\n".join(
        [
        "PROPAGATED NULL: requirements",
        "",
        "1. Fit a generative model in the RAW data space, across ALL subjects,",
        "   so that no subgroup structure exists by construction while the",
        "   first- and second-order statistics match the observed cohort.",
        "",
        "2. Write each surrogate cohort to disk in the real subject layout and",
        "   run it through the UNMODIFIED pipeline, redirecting only the data",
        "   path. Do not reimplement any transform. Keep identifiers, exclusion",
        "   manifests and covariate tables untouched; only the data is surrogate.",
        "",
        "3. VALIDATE THE FORWARD PATH BEFORE TRUSTING THE NULL. Regenerate real",
        "   subjects' features through the surrogate path and confirm they match",
        "   the originals to storage precision. Without this the null traverses a",
        "   different pipeline than the observed data and the comparison is void.",
        "",
        "4. Hold the training configuration, seed included, at the value the",
        "   observed statistic used. The observed statistic is one realization of",
        "   the same pipeline, and a null over realizations of it is the correct",
        "   comparison.",
        "",
        "5. RUN THE PAIRED CHECK. Re-run ~30 draws at identical seeds so each",
        "   pair shares a bit-identical surrogate cohort and differs only in",
        "   retraining. Decompose the null's variance. If the selected k flips",
        "   between paired re-runs, your null's spread is mostly retraining",
        "   nondeterminism rather than data variability: still a valid test, but",
        "   measuring something other than what you probably intend. In the",
        "   paper this was ~80% for one cohort and ~0% for the other, at the",
        "   same method; it is a property of the configuration, not the method.",
        "",
        "Cost: one full retraining per draw. 500 draws came to ~23 h at n=28",
        "and ~3.4 h at n=51 on one GPU. 200 draws would have returned the same",
        "verdict in both cohorts.",
        ]
        )

if __name__=="__main__":
    rng=np.random.default_rng(0)
    n,d=28,119
    spectrum=np.exp(-np.arange(d)/3.0)
    Z=rng.standard_normal((n,d))*np.sqrt(spectrum)
    Q,_=np.linalg.qr(rng.standard_normal((d,d)))
    X=Z@Q.T
    print(f"participation ratio: {participation_ratio(X):.2f} unnormalized, "
        f"{participation_ratio(X, normalize=True):.2f} on the sphere\n")
    print(ladder(X,n_draws=500).table())
    print()
    print("There is no structure in this data by construction. If the four "
        "p-values disagree, the disagreement is the artifact.")
