import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

PINK="#DDA7A5"
BROWN="#B88E8C"
LATTE="#966B6D"
COFFEE="#734F50"
BLACK="#000000"
PRIMARY=PINK
SECONDARY=BROWN
HIGHLIGHT=COFFEE
ALPHA=0.05
ARI_REAL=0.80
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
    for base in (rd, *rd.parents):
        c=base/"FM_20run_stability"
        if c.exists():
            return c
    return None

def _pairwise_ari(mat):
    n=len(mat)
    return float(np.mean([adjusted_rand_score(mat[i],mat[j]) for i,j in itertools.combinations(range(n),2)]))

def plot_pipeline_instability(results_dir=DEFAULT_RESULTS, stability_dir=None):
    _style()
    results_dir=Path(results_dir)
    stability_dir=Path(stability_dir) if stability_dir is not None else _find_stability(results_dir)
    if stability_dir is None or not stability_dir.exists():
        raise FileNotFoundError(f"FM_20run_stability not found near {results_dir}")
    fa_dir=results_dir/"fm_fixed_arch_control"
    if not fa_dir.exists():
        raise FileNotFoundError(f"fm_fixed_arch_control not found at {fa_dir}")
    run=pd.read_csv(stability_dir/"by_run"/"fm_silhouette_by_run.csv")
    perm=run["fm_permp_selected"].to_numpy(dtype=float)
    ksv=run["fm_k_sel_sil"].to_numpy(dtype=int)
    kfa=pd.read_csv(fa_dir/"fixed_arch_by_run.csv")["k_sel"].to_numpy(dtype=int)
    ab=pd.read_csv(results_dir/"ablation_table.csv").set_index("condition")
    canon=float(ab.loc["Full model","perm_p"])
    boot=float(pd.read_csv(results_dir/"bootstrap_stability.csv")["ari"].mean())
    svlab=pd.read_csv(stability_dir/"by_run"/"fm_labels_by_run.csv").set_index("run").to_numpy()
    sv_ari=_pairwise_ari(svlab)
    fal=[pd.read_csv(fa_dir/"by_run"/f"run{r}_labels.csv").set_index("Subject_Id")["Label"] for r in range(1,21)]
    common=sorted(set.intersection(*[set(s.index) for s in fal]))
    fa_ari=_pairwise_ari([s.loc[common].to_numpy() for s in fal])

    fig,ax=plt.subplots(1,3,figsize=(13.8,4.3))
    ps=np.sort(perm)
    n=len(ps)
    nsig=int(np.sum(ps<ALPHA))
    nsig_corr=int(np.sum(ps<ALPHA/n))       
    med=float(np.median(ps))
    cols=[HIGHLIGHT if v<ALPHA else PRIMARY for v in ps]
    ax[0].bar(np.arange(1,n+1),ps,color=cols,width=0.82,edgecolor="white",linewidth=0.4)
    ax[0].axhline(ALPHA,color=BLACK,ls="--",lw=1.1)
    ax[0].axhline(canon,color=LATTE,ls=":",lw=1.6)
    ax[0].text(n,ALPHA,f"  p={ALPHA:g}",color=BLACK,va="bottom",ha="right",fontsize=8)
    ax[0].text(0.4,canon,f"canonical p={canon:g}",color=BLACK,va="bottom",ha="left",fontsize=8)
    ax[0].text(0.98,0.97,f"{nsig} of {n} uncorrected; {nsig_corr} of {n} after correction\nmedian p = {med:.3f}",
               transform=ax[0].transAxes,ha="right",va="top",fontsize=8)
    ax[0].set_xlabel("run (sorted by permutation p)")
    ax[0].set_ylabel("covariance-preserving permutation p")
    ax[0].set_title("A  clustering significance (search-varying)",fontsize=10,loc="left")
    ax[0].margins(x=0.02)
    kvals=sorted(set(ksv.tolist())|set(kfa.tolist()))
    sc=[int(np.sum(ksv==k)) for k in kvals]
    fc=[int(np.sum(kfa==k)) for k in kvals]
    x=np.arange(len(kvals)); w=0.38
    ax[1].bar(x-w/2,sc,w,color=PRIMARY,edgecolor="white",linewidth=0.6,label="search-varying")
    ax[1].bar(x+w/2,fc,w,color=SECONDARY,edgecolor="white",linewidth=0.6,label="fixed-architecture")
    for xi,v in zip(x-w/2,sc): ax[1].text(xi,v,str(v),ha="center",va="bottom",fontsize=8)
    for xi,v in zip(x+w/2,fc): ax[1].text(xi,v,str(v),ha="center",va="bottom",fontsize=8)
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(k) for k in kvals])
    ax[1].set_xlabel("silhouette-selected k")
    ax[1].set_ylabel("number of runs")
    ax[1].set_title("B  selected k unstable under both protocols",fontsize=10,loc="left")
    ax[1].set_ylim(0,max(max(sc),max(fc))*1.20)
    ax[1].legend(frameon=False,fontsize=8,loc="upper right")
    labels=["within-checkpoint\nbootstrap","search-varying\ncross-run","fixed-arch\ncross-run"]
    vals=[boot,sv_ari,fa_ari]
    ax[2].bar(labels,vals,color=[SECONDARY,HIGHLIGHT,HIGHLIGHT],width=0.62,edgecolor="white",linewidth=0.6)
    ax[2].axhline(ARI_REAL,color=BLACK,ls="--",lw=1.2)
    ax[2].text(2.45,ARI_REAL,f"real subtypes ~{ARI_REAL:g}",color=BLACK,va="bottom",ha="right",fontsize=8)
    for xi,v in enumerate(vals): ax[2].text(xi,v,f"{v:.2f}",ha="center",va="bottom",fontsize=9)
    ax[2].set_ylabel("adjusted Rand index")
    ax[2].set_title("C  partitions do not reproduce",fontsize=10,loc="left")
    ax[2].set_ylim(0,1.0)
    fig.tight_layout()
    return fig

def plot_silhouette_vs_effrank(results_dir=DEFAULT_RESULTS):
    _style()
    results_dir=Path(results_dir)
    ab=pd.read_csv(results_dir/"ablation_table.csv").set_index("condition")
    order=["Untrained encoder","Mean pooling","Full model"]
    nice={"Untrained encoder":"untrained","Mean pooling":"mean-pooling","Full model":"trained"}
    ab=ab.loc[order]
    sil=ab["silhouette"].to_numpy(dtype=float)
    er=ab["eff_rank"].to_numpy(dtype=float)
    x=np.arange(len(order))
    w=0.38
    fig,axL=plt.subplots(figsize=(7.6,4.8))
    axR=axL.twinx()
    axR.spines["right"].set_visible(True)
    axR.spines["top"].set_visible(False)
    axR.grid(False)
    b1=axL.bar(x-w/2,sil,w,color=PRIMARY,edgecolor="white",linewidth=0.6,label="silhouette")
    b2=axR.bar(x+w/2,er,w,color=SECONDARY,edgecolor="white",linewidth=0.6,label="effective rank")
    for xi,v in zip(x-w/2,sil):
        axL.text(xi,v,f"{v:.3f}",ha="center",va="bottom",fontsize=8,color=BLACK)
    for xi,v in zip(x+w/2,er):
        axR.text(xi,v,f"{v:.2f}",ha="center",va="bottom",fontsize=8,color=BLACK)
    axL.set_xticks(x)
    axL.set_xticklabels([nice[o] for o in order])
    axL.set_ylabel("silhouette",color=BLACK)
    axR.set_ylabel("effective rank",color=BLACK)
    axL.tick_params(axis="y",colors=BLACK)
    axR.tick_params(axis="y",colors=BLACK)
    axL.set_ylim(0,max(sil)*1.25)
    axR.set_ylim(0,max(er)*1.25)
    fig.legend([b1,b2],["silhouette","effective rank"],loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(0.5,1.005))
    fig.tight_layout(rect=(0,0,1,0.95))
    return fig

def save_all(results_dir=DEFAULT_RESULTS, dpi=200):
    out=Path(results_dir)/"figures"
    out.mkdir(parents=True,exist_ok=True)
    f1=plot_pipeline_instability(results_dir)
    f1.savefig(out/"fig1_pipeline_instability.png",dpi=dpi,bbox_inches="tight")
    f2=plot_silhouette_vs_effrank(results_dir)
    f2.savefig(out/"fig2_silhouette_vs_effrank.png",dpi=dpi,bbox_inches="tight")
    return out/"fig1_pipeline_instability.png",out/"fig2_silhouette_vs_effrank.png"
if __name__=="__main__":
    p1,p2=save_all()
    print(f"wrote {p1}")
    print(f"wrote {p2}")
