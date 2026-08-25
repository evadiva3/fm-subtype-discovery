import json,glob,csv,time
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score,silhouette_score
import ov_common as ov
import nullcal as nc
OUT=ov.OUTDIR/"e11_consensus.json"
ROOT=ov.ROOT
N_BOOT=1000
K_RANGE=range(2,9)
SEED=20260821


def load_fixed_arch():
    fi=sorted(glob.glob(str(ROOT/"results/mdd/fixed_arch_control/runs/seed*.json")))
    l,ids=[],None
    for f in fi:
        j=json.loads(Path(f).read_text())
        if "labels" not in j:
            continue
        l.append(np.asarray(j["labels"],dtype=int))
        ids=ids or j.get("subject_ids")
    return(np.array(l)if l else None),ids,len(fi)


def load_search_vary():
    fi=sorted(glob.glob(str(ROOT/"results/mdd/multirun_v2/by_run/run*_labels.csv")),
        key=lambda p:int(Path(p).stem.split("_")[0][3:]))
    l,ids=[],None
    for f in fi:
        ro=list(csv.DictReader(open(f)))
        if ids is None:
            ids=[r["Subject_Id"]for r in ro]
        o={r["Subject_Id"]:int(r["Label"])for r in ro}
        l.append(np.array([o[s]for s in ids],dtype=int))
    return(np.array(l)if l else None),ids,len(fi)


def cooccurrence(L):
    R,n=L.shape
    C=np.zeros((n,n))
    for r in range(R):
        C+=(L[r][:,None]==L[r][None,:]).astype(float)
    return C/R


def consensus_partition(C,guard,k_range=K_RANGE):
    Dm=1.0-C
    np.fill_diagonal(Dm,0.0)
    Dm=np.clip((Dm+Dm.T)/2.0,0.0,None)
    Z=linkage(squareform(Dm,checks=False),method="average")
    b=(-np.inf,None,None)
    for k in k_range:
        lab=fcluster(Z,t=k,criterion="maxclust")
        if len(np.unique(lab))<2:
            continue
        if np.bincount(lab).min()>0 and np.bincount(lab)[np.bincount(lab)>0].min()<guard:
            continue
        s=float(silhouette_score(Dm,lab,metric="precomputed"))
        if s>b[0]:
            b=(s,int(k),lab)
    if b[1]is None:
        for k in k_range:
            lab=fcluster(Z,t=k,criterion="maxclust")
            if len(np.unique(lab))<2:
                continue
            s=float(silhouette_score(Dm,lab,metric="precomputed"))
            if s>b[0]:
                b=(s,int(k),lab)
        return b[2],b[1],b[0],False
    return b[2],b[1],b[0],True


def analyse(name,L,ids,n_files):
    R,n=L.shape
    g=nc.min_cluster_guard(n)
    C=cooccurrence(L)
    c,kc,sc,go=consensus_partition(C,g)
    ruvc=[float(adjusted_rand_score(L[r],c))for r in range(R)]
    p=[float(adjusted_rand_score(L[i],L[j]))
        for i in range(R)for j in range(i+1,R)]
    rng=np.random.default_rng(SEED)
    b=[]
    bk=[]
    for _ in range(N_BOOT):
        idx=rng.integers(0,R,size=R)
        Cb=cooccurrence(L[idx])
        cb,kb,_,_=consensus_partition(Cb,g)
        b.append(float(adjusted_rand_score(cb,c)))
        bk.append(int(kb))
    b=np.array(b)
    p=np.array(p)
    rvc=np.array(ruvc)
    iu=np.triu_indices(n,1)
    off=C[iu]
    return dict(
        protocol=name,n_runs=R,n_files_found=n_files,n_subjects=n,
        guard=g,guard_satisfied=go,
        consensus_k=kc,consensus_silhouette=sc,
        consensus_sizes=np.bincount(c)[np.bincount(c)>0].tolist(),
        consensus_labels=c.tolist(),subject_ids=ids,
        run_k=[int(len(np.unique(L[r])))for r in range(R)],
        cooccurrence_mean=float(off.mean()),cooccurrence_sd=float(off.std(ddof=1)),
        cooccurrence_frac_below_0_25=float((off<0.25).mean()),
        cooccurrence_frac_above_0_75=float((off>0.75).mean()),
        ari_run_vs_consensus_mean=float(rvc.mean()),
        ari_run_vs_consensus_sd=float(rvc.std(ddof=1)),
        ari_run_vs_consensus_min=float(rvc.min()),
        ari_run_vs_consensus_max=float(rvc.max()),
        ari_run_vs_consensus=rvc.tolist(),
        ari_pairwise_runs_mean=float(p.mean()),
        ari_pairwise_runs_sd=float(p.std(ddof=1)),
        ari_pairwise_runs_min=float(p.min()),
        ari_pairwise_runs_max=float(p.max()),
        n_pairs=int(p.size),
        bootstrap_n=N_BOOT,
        ari_bootstrap_consensus_mean=float(b.mean()),
        ari_bootstrap_consensus_sd=float(b.std(ddof=1)),
        ari_bootstrap_consensus_q05=float(np.quantile(b,0.05)),
        ari_bootstrap_consensus_q50=float(np.quantile(b,0.50)),
        ari_bootstrap_consensus_q95=float(np.quantile(b,0.95)),
        bootstrap_k_counts={int(k):int((np.array(bk)==k).sum())
        for k in sorted(set(bk))},
        consensus_more_stable=bool(b.mean()>p.mean()),
        stability_gain=float(b.mean()-p.mean()))


def main():
    t0=time.time()
    s=ov.load_state(OUT,"E11",dict(n_bootstrap=N_BOOT,
        k_range=list(K_RANGE),seed=SEED))
    l={"fixed_arch":load_fixed_arch,"search_vary":load_search_vary}
    for n,fn in l.items():
        if n in s["cells"]:
            print("  skip %s"%n,flush=True)
            continue
        try:
            L,ids,nf=fn()
        except Exception as e:
            s["cells"][n]=dict(protocol=n,skipped=True,reason=str(e))
            print("[E11] %s SKIPPED: %s"%(n,e),flush=True)
            continue
        if L is None or L.size==0:
            s["cells"][n]=dict(protocol=n,skipped=True,
                reason="no label vectors on disk")
            print("[E11] %s SKIPPED: no label vectors on disk"%n,flush=True)
            continue
        c=analyse(n,L,ids,nf)
        s["cells"][n]=c
        ov.save_state(OUT,s)
        print("[E11] %-12s runs=%d n=%d  consensus k=%d sizes=%s  "
            "ARI run-vs-consensus %.4f+-%.4f  pairwise runs %.4f+-%.4f  "
            "bootstrap-consensus %.4f+-%.4f  gain=%+.4f"
            %(n,c["n_runs"],c["n_subjects"],c["consensus_k"],
            c["consensus_sizes"],c["ari_run_vs_consensus_mean"],
            c["ari_run_vs_consensus_sd"],c["ari_pairwise_runs_mean"],
            c["ari_pairwise_runs_sd"],
            c["ari_bootstrap_consensus_mean"],
            c["ari_bootstrap_consensus_sd"],c["stability_gain"]),
            flush=True)
    fm=ROOT/"results/fm_fixed_arch_control"
    if "fm_note" not in s["cells"]:
        hl=bool(glob.glob(str(fm/"**/*labels*"),recursive=True))
        s["cells"]["fm_note"]=dict(
            protocol="fm_fixed_arch",skipped=True,
            reason=("fibromyalgia fixed-architecture control stores only per-run "
            "summary statistics (run, seed, k_selected, silhouette, perm_p) "
            "in fixed_arch_control_runs.csv; no per-run label vectors and no "
            "per-run embeddings are on disk, so no consensus can be built "
            "for that cohort without retraining, which is out of scope."),
            labels_found=hl)
        print("[E11] fm_fixed_arch SKIPPED: no per-run label vectors on disk",
            flush=True)
    s["wall_s"]=round(time.time()-t0,1)
    ov.save_state(OUT,s)
    print("E11 done in %.1fs"%(time.time()-t0))

if __name__=="__main__":
    main()
