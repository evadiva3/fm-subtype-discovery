import os,sys,json,time,warnings,traceback
for _v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nullcal as nc
ROOT=Path(__file__).resolve().parents[3]
OUTDIR=ROOT/"results"/"overnight"
D=119
N_DRAWS=200
N_INIT=5
K_GRID=(2,3,4,5,6)
ALPHA=0.05
TOL=0.05
TUNE_M=20
def factor(d,lam,rng):
    lam=np.asarray(lam,dtype=np.float64)
    Q,_=np.linalg.qr(rng.standard_normal((d,d)))
    return np.sqrt(lam)[:,None]*Q.T


def gen(n,d,lam,seed):
    rng=np.random.default_rng(seed)
    X=rng.standard_normal((n,d))@factor(d,lam,rng)
    return nc.l2_normalize(X)


def mean_pr_lam(n,d,lam,seed=987654,m=TUNE_M):
    return float(np.mean([nc.participation_ratio(gen(n,d,lam,seed+31*i))
        for i in range(m)]))


def spec_exp(d,tau):
    return np.ones(d)if np.isinf(tau)else np.exp(-np.arange(d)/tau)


def spec_power(d,a):
    return(np.arange(d)+1.0)**(-a)


def spec_two_block(d,m,eps):
    lam=np.full(d,float(eps))
    lam[:int(m)]=1.0
    return lam


def spec_one_spike(d,A):
    lam=np.ones(d)
    lam[0]=float(A)
    return lam


def spec_few_spikes(d,A,k=5):
    lam=np.ones(d)
    lam[:k]=float(A)
    return lam


def bisect_param(n,d,make_lam,lo,hi,target,tol=TOL,iters=60,
    seed=987654,m=TUNE_M):
    f=lambda p:mean_pr_lam(n,d,make_lam(p),seed,m)
    pl,ph=f(lo),f(hi)
    i=ph>pl
    av,bv=(pl,ph)if i else(ph,pl)
    if target<=av:
        return(lo if i else hi),av,False,"target below family floor"
    if target>=bv:
        return(hi if i else lo),bv,False,"target above family ceiling"
    g=lo>0 and hi>0
    cur,pm=hi,ph
    for _ in range(iters):
        cur=float(np.sqrt(lo*hi))if g else 0.5*(lo+hi)
        pm=f(cur)
        if abs(pm-target)/target<=tol:
            return cur,pm,True,""
        b=pm<target
        if b==i:
            lo=cur
        else:
            hi=cur
        if g and hi/lo<1.0000001:
            break
        if not g and abs(hi-lo)<1e-12:
            break
    ok=abs(pm-target)/target<=tol
    return cur,pm,ok,"" if ok else "bisection stalled"


def tune_two_block(n,d,target,m_candidates=range(1,14)):
    b=None
    for mm in m_candidates:
        p,pr,ok,no=bisect_param(
            n,d,lambda e,mm=mm:spec_two_block(d,mm,e),1e-6,1.0,target)
        if ok:
            return dict(m=int(mm),eps=float(p)),pr,True,""
        err=abs(pr-target)
        if b is None or err<b[0]:
            b=(err,dict(m=int(mm),eps=float(p)),pr,no)
    return b[1],b[2],False,b[3]or "no (m, eps) within tolerance"


def mc_worker(args):
    n,d,lam,idx,s,nd,kg=args
    try:
        Xn=gen(n,d,lam,s)
        pr=nc.participation_ratio(Xn)
        ms=nc.min_cluster_guard(n)
        obs=nc.best_silhouette(Xn,k_grid=kg,min_size=ms,
            n_init=N_INIT,random_state=idx)
        if obs[1]is None:
            return{"skip":"no k passes guard on observed"}
        kw=dict(k_grid=kg,n_draws=nd,min_size=ms,n_init=N_INIT,
            seed=s+7919,observed=(obs[0],obs[1]))
        rM=nc.null_test(Xn,reproject=False,match_selection=False,**kw)
        rC=nc.null_test(Xn,reproject=True,match_selection=True,**kw)
        return dict(pr=float(pr),pM=float(rM.p),pC=float(rC.p),k=int(obs[1]),
            obs=float(obs[0]),
            nmM=float(rM.null_mean),nsM=float(rM.null_sd),
            nmC=float(rC.null_mean),nsC=float(rC.null_sd),
            degC=int(rC.n_degenerate))
    except Exception:
        return{"error":traceback.format_exc()}


def summarize_mc(res,wall,**extra):
    e=[r["error"]for r in res if "error" in r]
    s=[r for r in res if "skip" in r]
    g=[r for r in res if "pM" in r]
    if not g:
        d=dict(failed=True,n_errors=len(e),n_skipped=len(s),
            error_sample=e[0]if e else None,wall_s=round(wall,1))
        d.update(extra)
        return d
    pM=np.array([r["pM"]for r in g])
    pC=np.array([r["pC"]for r in g])
    pr=np.array([r["pr"]for r in g])
    cM,cC,N=int((pM<ALPHA).sum()),int((pC<ALPHA).sum()),len(g)
    out=dict(
        pr_mean=float(pr.mean()),pr_sd=float(pr.std(ddof=1)),
        n_datasets=N,n_skipped=len(s),n_errors=len(e),
        rej_M=cM/N,count_M=cM,ci_M=list(nc._wilson(cM,N)),
        rej_C=cC/N,count_C=cC,ci_C=list(nc._wilson(cC,N)),
        mean_pM=float(pM.mean()),mean_pC=float(pC.mean()),
        ks_M=float(nc._ks_uniform(pM)),ks_C=float(nc._ks_uniform(pC)),
        null_mean_M=float(np.mean([r["nmM"]for r in g])),
        null_sd_M=float(np.mean([r["nsM"]for r in g])),
        null_mean_C=float(np.mean([r["nmC"]for r in g])),
        null_sd_C=float(np.mean([r["nsC"]for r in g])),
        obs_mean=float(np.mean([r["obs"]for r in g])),
        M_only=int(((pM<ALPHA)&(pC>=ALPHA)).sum()),
        C_only=int(((pC<ALPHA)&(pM>=ALPHA)).sum()),
        degenerate_C=int(np.sum([r["degC"]for r in g])),
        k_counts={int(k):int(sum(1 for r in g if r["k"]==k))
        for k in sorted({r["k"]for r in g})},
        wall_s=round(wall,1),failed=False)
    out.update(extra)
    return out


def summarize_single(pvals,wall,**extra):
    p=np.asarray(pvals,dtype=float)
    c,N=int((p<ALPHA).sum()),p.size
    out=dict(n_datasets=N,rej=c/N,count=c,ci=list(nc._wilson(c,N)),
        mean_p=float(p.mean()),ks=float(nc._ks_uniform(p)),
        wall_s=round(wall,1))
    out.update(extra)
    return out


def load_state(path,experiment,config):
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return{"experiment":experiment,"config":config,"cells":{}}


def save_state(path,state):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(state,indent=1))


def write_csv(path,cells,cols):
    r=[",".join(cols)]
    for key,c in cells.items():
        if c.get("failed"):
            continue
        rec=dict(c)
        rec["cell"]=key
        for a,b in(("ci_M","ci_M"),("ci_C","ci_C")):
            if a in rec:
                rec[b+"_lo"],rec[b+"_hi"]=rec[a][0],rec[a][1]
        va=[]
        for col in cols:
            v=rec.get(col,"")
            va.append(round(v,5)if isinstance(v,float)else v)
        r.append(",".join(str(v)for v in va))
    Path(path).write_text("\n".join(r)+"\n")


def elapsed_budget():
    f=OUTDIR/".t0"
    if not f.exists():
        return 0.0
    import datetime as _dt
    t0=_dt.datetime.strptime(f.read_text().strip(),"%Y-%m-%d %H:%M:%S")
    return(_dt.datetime.now()-t0).total_seconds()/3600.0
