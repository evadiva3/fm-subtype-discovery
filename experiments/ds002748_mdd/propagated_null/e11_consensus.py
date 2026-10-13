import json,glob,csv,time
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score,silhouette_score
import ov_common as ov
import nullcal as nc
OUT=ov.OUTDIR/'e11_consensus.json'
ROOT=ov.ROOT
N_BOOT=1000
K_RANGE=range(2,9)
SEED=20260821

def load_fixed_arch():
    C=sorted(glob.glob(str(ROOT/'results/mdd/fixed_arch_control/runs/seed*.json')))
    A,ids= ([],None)
    for D in C:
        B=json.loads(Path(D).read_text())
        if'labels'not in B:
            continue
        A.append(np.asarray(B['labels'],dtype=int))
        ids=ids or B.get('subject_ids')
    return (np.array(A)if A else None,ids,len(C))

def load_search_vary():
    B=sorted(glob.glob(str(ROOT/'results/mdd/multirun_v2/by_run/run*_labels.csv')),key=lambda p:int(Path(p).stem.split('_')[0][3:]))
    A,ids= ([],None)
    for D in B:
        C=list(csv.DictReader(open(D)))
        if ids is None:
            ids= [A['Subject_Id']for A in C]
        E= {A['Subject_Id']:int(A['Label'])for A in C}
        A.append(np.array([E[A]for A in ids],dtype=int))
    return (np.array(A)if A else None,ids,len(B))

def cooccurrence(L):
    A,B=L.shape
    C=np.zeros((B,B))
    for D in range(A):
        C+= (L[D][:,None]==L[D][None,:]).astype(float)
    return C/A

def consensus_partition(C,guard,k_range=K_RANGE):
    D=1.0-C
    np.fill_diagonal(D,0.0)
    D=np.clip((D+D.T)/2.0,0.0,None)
    G=linkage(squareform(D,checks=False),method='average')
    A= (-np.inf,None,None)
    for E in k_range:
        B=fcluster(G,t=E,criterion='maxclust')
        if len(np.unique(B))<2:
            continue
        if np.bincount(B)[np.bincount(B)>0].min()<guard:
            continue
        F=float(silhouette_score(D,B,metric='precomputed'))
        if F>A[0]:
            A= (F,int(E),B)
    if A[1]is None:
        for E in k_range:
            B=fcluster(G,t=E,criterion='maxclust')
            if len(np.unique(B))<2:
                continue
            F=float(silhouette_score(D,B,metric='precomputed'))
            if F>A[0]:
                A= (F,int(E),B)
        return (A[2],A[1],A[0],False)
    return (A[2],A[1],A[0],True)

def analyse(name,L,ids,n_files):
    B,G=L.shape
    H=nc.min_cluster_guard(G)
    C=cooccurrence(L)
    D,K,M,N=consensus_partition(C,H)
    O= [float(adjusted_rand_score(L[A],D))for A in range(B)]
    p= [float(adjusted_rand_score(L[A],L[C]))for A in range(B)for C in range(A+1,B)]
    P=np.random.default_rng(SEED)
    A= []
    I= []
    V=0
    for J in range(N_BOOT):
        Q=P.integers(0,B,size=B)
        R=cooccurrence(L[Q])
        S,T,J,F0=consensus_partition(R,H)
        V+=int(not F0)
        A.append(float(adjusted_rand_score(S,D)))
        I.append(int(T))
    A=np.array(A)
    p=np.array(p)
    E=np.array(O)
    U=np.triu_indices(G,1)
    F=C[U]
    return dict(protocol=name,n_runs=B,n_files_found=n_files,n_subjects=G,guard=H,guard_satisfied=N,consensus_k=K,consensus_silhouette=M,consensus_sizes=np.bincount(D)[np.bincount(D)>0].tolist(),consensus_labels=D.tolist(),subject_ids=ids,run_k=[int(len(np.unique(L[A])))for A in range(B)],cooccurrence_mean=float(F.mean()),cooccurrence_sd=float(F.std(ddof=1)),cooccurrence_frac_below_0_25=float((F<0.25).mean()),cooccurrence_frac_above_0_75=float((F>0.75).mean()),ari_run_vs_consensus_mean=float(E.mean()),ari_run_vs_consensus_sd=float(E.std(ddof=1)),ari_run_vs_consensus_min=float(E.min()),ari_run_vs_consensus_max=float(E.max()),ari_run_vs_consensus=E.tolist(),ari_pairwise_runs_mean=float(p.mean()),ari_pairwise_runs_sd=float(p.std(ddof=1)),ari_pairwise_runs_min=float(p.min()),ari_pairwise_runs_max=float(p.max()),n_pairs=int(p.size),bootstrap_n=N_BOOT,bootstrap_guard_failures=V,ari_bootstrap_consensus_mean=float(A.mean()),ari_bootstrap_consensus_sd=float(A.std(ddof=1)),ari_bootstrap_consensus_q05=float(np.quantile(A,0.05)),ari_bootstrap_consensus_q50=float(np.quantile(A,0.5)),ari_bootstrap_consensus_q95=float(np.quantile(A,0.95)),bootstrap_k_counts={int(A):int((np.array(I)==A).sum())for A in sorted(set(I))},consensus_more_stable=bool(A.mean()>p.mean()),stability_gain=float(A.mean()-p.mean()))

def main():
    D=time.time()
    C=ov.load_state(OUT,'E11',dict(n_bootstrap=N_BOOT,k_range=list(K_RANGE),seed=SEED))
    F= {'fixed_arch':load_fixed_arch,'search_vary':load_search_vary}
    for B,G in F.items():
        if B in C['cells']:
            print('  skip %s'%B,flush=True)
            continue
        try:
            L,ids,H=G()
        except Exception as E:
            C['cells'][B]=dict(protocol=B,skipped=True,reason=str(E))
            print('[E11] %s SKIPPED: %s'% (B,E),flush=True)
            continue
        if L is None or L.size==0:
            C['cells'][B]=dict(protocol=B,skipped=True,reason='no label vectors on disk')
            print('[E11] %s SKIPPED: no label vectors on disk'%B,flush=True)
            continue
        A=analyse(B,L,ids,H)
        C['cells'][B]=A
        ov.save_state(OUT,C)
        print('[E11] %-12s runs=%d n=%d  consensus k=%d sizes=%s  ARI run-vs-consensus %.4f+-%.4f  pairwise runs %.4f+-%.4f  bootstrap-consensus %.4f+-%.4f  gain=%+.4f'% (B,A['n_runs'],A['n_subjects'],A['consensus_k'],A['consensus_sizes'],A['ari_run_vs_consensus_mean'],A['ari_run_vs_consensus_sd'],A['ari_pairwise_runs_mean'],A['ari_pairwise_runs_sd'],A['ari_bootstrap_consensus_mean'],A['ari_bootstrap_consensus_sd'],A['stability_gain']),flush=True)
    I=ROOT/'results/fm_fixed_arch_control'
    if'fm_note'not in C['cells']:
        J=bool(glob.glob(str(I/'**/*labels*'),recursive=True))
        C['cells']['fm_note']=dict(protocol='fm_fixed_arch',skipped=True,reason='fibromyalgia fixed-architecture control stores only per-run summary statistics (run, seed, k_selected, silhouette, perm_p) in fixed_arch_control_runs.csv; no per-run label vectors and no per-run embeddings are on disk, so no consensus can be built for that cohort without retraining, which is out of scope.',labels_found=J)
        print('[E11] fm_fixed_arch SKIPPED: no per-run label vectors on disk',flush=True)
    C['wall_s']=round(time.time()-D,1)
    ov.save_state(OUT,C)
    print('E11 done in %.1fs'% (time.time()-D))
if __name__=='__main__':
    main()
