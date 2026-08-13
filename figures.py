import json,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from analysis.evaluate import cluster_evaluate

PINK="#DDA7A5"
BROWN="#B88E8C"
LATTE="#966B6D"
COFFEE="#734F50"
BLACK="#000000"
PRIMARY=PINK
SECONDARY=BROWN
HIGHLIGHT=COFFEE
ALPHA=0.05
_HERE=Path(__file__).resolve().parent
DEFAULT_RESULTS=_HERE/"results"

def _style():
    plt.rcParams.update({
        "font.family":"serif",
        "font.serif":["Palatino","Palatino Linotype","Georgia","DejaVu Serif"],
        "axes.spines.top":False,
        "axes.spines.right":False,
        "axes.edgecolor":"#3a3a3a",
        "axes.linewidth":0.8,
        "axes.grid":True,
        "axes.grid.axis":"y",
        "axes.axisbelow":True,
        "grid.color":"#c9c9c9",
        "grid.linewidth":0.5,
        "grid.alpha":0.5,
        "xtick.color":"#3a3a3a",
        "ytick.color":"#3a3a3a",
    })

def _find_stability(results_dir):
    rd=Path(results_dir).resolve()
    for base in(rd,*rd.parents):
        c=base/"FM_20run_stability"
        if c.exists():
            return c
    return None

def _recompute_null_draws(X,k,mode,n_perm,seed=42,n_init=20,label=None):
    if mode not in("old","B"):
        raise ValueError(f"unknown null mode {mode!r}; expected 'old' or 'B'")
    X=np.asarray(X,dtype=float)
    Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
    lab=(KMeans(n_clusters=k,n_init=n_init,random_state=seed).fit_predict(Xn)
         if label is None else np.asarray(label))
    real=float(silhouette_score(Xn,lab))
    mu=Xn.mean(axis=0); cov=np.cov(Xn,rowvar=False)
    rng=np.random.default_rng(seed)
    ev=cluster_evaluate()
    nulls=np.empty(n_perm); c=0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_perm):
            d=rng.multivariate_normal(mu,cov,size=X.shape[0],method="svd")
            if mode=="B":
                d=d/(np.linalg.norm(d,axis=1,keepdims=True)+1e-8)
                s=ev._select_best_silhouette(d,random_state=i)
            else:
                dl=KMeans(n_clusters=k,n_init=n_init,random_state=i).fit_predict(d)
                s=silhouette_score(d,dl)
            nulls[i]=s
            if s>=real: c+=1
    return real,nulls,(c+1)/(n_perm+1)

def _null_draws_k4(results_dir,force=False):
    results_dir=Path(results_dir)
    ref=json.load(open(results_dir/"null_corrected"/"phase4_canonical_k4.json"))
    cache=results_dir/"figures"/"_fig1a_null_draws_k4.npz"
    if cache.exists() and not force:
        z=np.load(cache)
        return float(z["real"]),z["mis"],z["cor"],ref
    emb=results_dir.parent/"data"/"outputs"/"trained_fm_embeddings.npy"
    X=np.load(emb)
    k=int(ref["old"]["k"]); nperm=int(ref["old"]["n_perm"])
    real_m,mis,p_m=_recompute_null_draws(X,k,"old",nperm)
    real_c,cor,p_c=_recompute_null_draws(X,k,"B",nperm)
    for tag,real,nulls,p in(("old",real_m,mis,p_m),("B",real_c,cor,p_c)):
        r=ref[tag]
        assert abs(real-r["real"])<1e-6,f"{tag} real drift {real} vs {r['real']}"
        assert abs(nulls.mean()-r["null_mean"])<1e-6,f"{tag} mean drift"
        assert abs(nulls.std()-r["null_std"])<1e-6,f"{tag} std drift"
        assert abs(p-r["p"])<1e-4,f"{tag} p drift {p} vs {r['p']}"
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,real=real_m,mis=mis,cor=cor)
    return real_m,mis,cor,ref

def plot_null_correction(results_dir=DEFAULT_RESULTS,stability_dir=None):
    _style()
    results_dir=Path(results_dir)
    stability_dir=Path(stability_dir) if stability_dir is not None else _find_stability(results_dir)
    if stability_dir is None or not stability_dir.exists():
        raise FileNotFoundError(f"FM_20run_stability not found near {results_dir}")
    real,mis,cor,ref=_null_draws_k4(results_dir)
    mis_mean=float(ref["old"]["null_mean"]); mis_p=float(ref["old"]["p"])
    cor_mean=float(ref["B"]["null_mean"]); cor_p=float(ref["B"]["p"])
    fig,ax=plt.subplots(1,3,figsize=(14.2,4.4))
    a=ax[0]
    lo=min(mis.min(),cor.min()); hi=max(real,mis.max(),cor.max())
    xs=np.linspace(lo-0.01,hi+0.01,500)
    kmis=stats.gaussian_kde(mis)(xs); kcor=stats.gaussian_kde(cor)(xs)
    a.fill_between(xs,kcor,color=PRIMARY,alpha=0.30,zorder=1)
    a.plot(xs,kcor,color=PRIMARY,lw=2.0,ls="-",zorder=3,label="corrected null")
    a.fill_between(xs,kmis,color=SECONDARY,alpha=0.15,hatch="////",edgecolor=SECONDARY,lw=0,zorder=1)
    a.plot(xs,kmis,color=SECONDARY,lw=2.0,ls="--",zorder=3,label="misspecified null")
    top=max(kmis.max(),kcor.max())
    a.axvline(real,color=HIGHLIGHT,lw=2.0,zorder=4)
    a.text(real,top*1.02,f"observed={real:.3f}\n(identical under both nulls)",
           color=HIGHLIGHT,ha="center",va="bottom",fontsize=8)
    a.annotate(f"mean {mis_mean:.3f}\np={mis_p:.3f}",xy=(mis_mean,0),
               xytext=(mis_mean-0.006,top*0.62),color=LATTE,fontsize=8,ha="right",
               arrowprops=dict(arrowstyle="-",color=SECONDARY,lw=0.8))
    a.annotate(f"mean {cor_mean:.3f}\np={cor_p:.3f}",xy=(cor_mean,0),
               xytext=(cor_mean+0.004,top*0.82),color=BLACK,fontsize=8,ha="left",
               arrowprops=dict(arrowstyle="-",color=PRIMARY,lw=0.8))
    a.set_ylim(0,top*1.30)
    a.set_xlabel("null silhouette (k=4, canonical checkpoint)")
    a.set_ylabel("density")
    a.set_title("A  the null moved, the data did not",fontsize=10,loc="left")
    a.legend(frameon=False,fontsize=8,loc="upper left")
    b=ax[1]
    run=pd.read_csv(stability_dir/"by_run"/"fm_silhouette_by_run.csv")
    old=run.set_index("run")["fm_permp_selected"].astype(float)
    c20=results_dir/"null_corrected"/"phase5_2_fm20run_corrected_20000.csv"
    if c20.exists():
        cdf=pd.read_csv(c20)
        cdf["run"]=cdf["run"].astype(int)
        corr=cdf.set_index("run")["p_20000"].astype(float)
        n_draws=20000
    else:
        cdf=pd.read_csv(results_dir/"null_corrected"/"phase5_2_fm20run_corrected.csv")
        corr=cdf.set_index("run")["new_p"].astype(float)
        n_draws=1000
        warnings.warn(f"{c20.name} absent; panels B/C fall back to 1,000 draws, which "
                      f"contradicts the Figure 1 caption")
    runs=sorted(set(old.index)&set(corr.index))
    sig=[r for r in runs if old[r]<ALPHA]
    for r in runs:
        hl=r in sig
        b.plot([0,1],[old[r],corr[r]],
                color=HIGHLIGHT if hl else PRIMARY,
                lw=1.8 if hl else 0.9,alpha=1.0 if hl else 0.55,
                marker="o",ms=4 if hl else 2.5,zorder=3 if hl else 2)
        if hl:
            b.text(-0.03,old[r],f"run {r}",color=HIGHLIGHT,ha="right",va="center",fontsize=7)
    b.axhline(ALPHA,color=BLACK,ls="--",lw=1.1)
    b.text(1.0,ALPHA,f" p={ALPHA:g}",color=BLACK,ha="left",va="bottom",fontsize=8)
    b.set_xlim(-0.28,1.28); b.set_xticks([0,1])
    b.set_xticklabels(["misspecified\nnull","corrected\nnull"])
    b.set_ylabel(f"permutation p (selected k, {n_draws:,} draws)")
    b.set_title("B  every p-value rose",fontsize=10,loc="left")
    b.set_ylim(0,max(old.max(),corr.max())*1.05)
    b.grid(axis="x",visible=False)
    c=ax[2]
    oc=np.sort(old.loc[runs].to_numpy()); cc=np.sort(corr.loc[runs].to_numpy())
    n=len(cc); y=np.arange(1,n+1)/n
    c.plot([0,1],[0,1],color="#8a8a8a",ls=":",lw=1.2,label="Uniform(0,1)")
    c.step(np.concatenate([[0],oc,[1]]),np.concatenate([[0],y,[1]]),where="post",
           color=SECONDARY,ls="--",lw=1.8,label="uncorrected")
    c.step(np.concatenate([[0],cc,[1]]),np.concatenate([[0],y,[1]]),where="post",
           color=PRIMARY,ls="-",lw=1.8,label="corrected")
    ks_c=stats.kstest(cc,"uniform")
    c.text(0.02,0.97,f"corrected vs Uniform(0,1)\nKS D={ks_c.statistic:.3f},  p={ks_c.pvalue:.3f}",
           transform=c.transAxes,ha="left",va="top",fontsize=8)
    c.set_xlim(0,1); c.set_ylim(0,1)
    c.set_xlabel(f"permutation p ({n_draws:,} draws)"); c.set_ylabel("empirical CDF")
    c.set_title("C  the corrected null behaves as a valid null",fontsize=10,loc="left")
    c.legend(frameon=False,fontsize=8,loc="lower right")

    fig.tight_layout()
    return fig

def plot_silhouette_vs_participation(results_dir=DEFAULT_RESULTS):
    _style()
    results_dir=Path(results_dir)
    ab=pd.read_csv(results_dir/"ablation_table.csv").set_index("condition")
    order=["Untrained encoder","Mean pooling","Full model"]
    nice={"Untrained encoder":"untrained","Mean pooling":"mean-pooling","Full model":"trained"}
    ab=ab.loc[order]
    sil=ab["silhouette"].to_numpy(dtype=float)
    pr=ab["eff_rank"].to_numpy(dtype=float)
    x=np.arange(len(order)); w=0.38
    fig,axL=plt.subplots(figsize=(7.6,4.8))
    axR=axL.twinx()
    axR.spines["right"].set_visible(True)
    axR.spines["top"].set_visible(False)
    axR.grid(False)
    b1=axL.bar(x-w/2,sil,w,color=PRIMARY,edgecolor="white",linewidth=0.6,label="silhouette")
    b2=axR.bar(x+w/2,pr,w,color=SECONDARY,edgecolor="white",linewidth=0.6,
               hatch="////",label="participation ratio")
    for xi,v in zip(x-w/2,sil):
        axL.text(xi,v,f"{v:.3f}",ha="center",va="bottom",fontsize=8,color=BLACK)
    for xi,v in zip(x+w/2,pr):
        axR.text(xi,v,f"{v:.2f}",ha="center",va="bottom",fontsize=8,color=BLACK)
    axL.set_xticks(x)
    axL.set_xticklabels([nice[o] for o in order])
    axL.set_ylabel("silhouette",color=BLACK)
    axR.set_ylabel("participation ratio",color=BLACK)
    axL.tick_params(axis="y",colors=BLACK)
    axR.tick_params(axis="y",colors=BLACK)
    axL.set_ylim(0,max(sil)*1.25)
    axR.set_ylim(0,max(pr)*1.25)
    fig.legend([b1,b2],["silhouette","participation ratio"],loc="upper center",ncol=2,
               frameon=False,bbox_to_anchor=(0.5,1.005))
    fig.tight_layout(rect=(0,0,1,0.95))
    return fig

def save_all(results_dir=DEFAULT_RESULTS,dpi=200):
    out=Path(results_dir)/"figures"
    out.mkdir(parents=True,exist_ok=True)
    written=[]
    f1=plot_null_correction(results_dir)
    p1=out/"fig1_null_correction.png"
    f1.savefig(p1,dpi=dpi,bbox_inches="tight")
    written.append(p1)
    f3=plot_silhouette_vs_participation(results_dir)
    p3=out/"fig3_silhouette_vs_participation.png"
    f3.savefig(p3,dpi=dpi,bbox_inches="tight")
    written.append(p3)
    p2=out/"fig2_reproducibility_crosscohort.png"
    if not p2.exists():
        print(f"{p2.name} absent; run results/figures/make_fig2_crosscohort.py")
    return tuple(written)

if __name__=="__main__":
    for p in save_all():
        print(f"wrote {p}")
