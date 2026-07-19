import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parent.parent
for d in (root, root/"src", root/"models", root/"analysis", root/"preprocessing"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from config import config
from subject_filter import get_included_subjects
from baselines import baselines
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
fmridir=root.parent/"fmriprep_output"
k=3
conds=[c.replace(" ", "") for c in config.conditions]

def _guard(label):
    n=len(label)
    mn=max(config.minClusterSizeFloor, round(config.minClusterSizeFraction*n))
    _, cnt=np.unique(label, return_counts=True)
    return bool((cnt>=mn).all())
def _perm(X, label, seed=None):
    seed=config.randomSeed if seed is None else seed
    X=np.asarray(X, dtype=float)
    kk=len(np.unique(label))
    real=silhouette_score(X, label)
    n=len(X)
    mu=X.mean(0); Xc=X-mu; sc=np.sqrt(max(n-1, 1))
    rng=np.random.default_rng(seed)
    c=0
    for i in range(config.nPermutations):
        nul=mu+(rng.standard_normal((n, n))@Xc)/sc
        lab=KMeans(n_clusters=kk, n_init=config.kmeansNInit, random_state=i).fit_predict(nul)
        if silhouette_score(nul, lab)>real:
            c+=1
    return c/config.nPermutations
def _acc_perm(X, y, acc, seed=None):
    seed=config.randomSeed if seed is None else seed
    rng=np.random.default_rng(seed)
    c=0
    for i in range(config.nPermutations):
        yp=rng.permutation(y)
        if cross_val_score(SVC(), X, yp, cv=config.svmCvFolds).mean()>=acc:
            c+=1
    return (c+1)/(config.nPermutations+1)
def load_mean_fc(sid):
    base=Path(config.subjectDataFolder)/sid
    mats=[]
    for c in conds:
        fp=base/f"{sid}_FCMatrixCondition{c}.npy"
        if not fp.exists():
            return None
        mats.append(np.load(fp))
    return np.mean(np.stack(mats, axis=0), axis=0)
def assemble():
    clin=pd.read_csv(config.clinicalCsv)
    clin["subject_id"]=clin["subject_id"].astype(str)
    grp=clin.set_index("subject_id")["group"].to_dict()
    inc=get_included_subjects()
    fcs, labs, ids, drop=[], [], [], []
    for sid in inc:
        g=grp.get(sid)
        if g not in ("FM", "HC"):
            drop.append((sid, f"group={g!r}"))
            continue
        fc=load_mean_fc(sid)
        if fc is None:
            drop.append((sid, "no FC npy"))
            continue
        fcs.append(fc)
        labs.append(0 if g=="FM" else 1)
        ids.append(sid)
    fcarr=np.stack(fcs, axis=0)
    labs=np.asarray(labs)
    nfm=int((labs==0).sum())
    nhc=int((labs==1).sum())
    print(f"cohort {len(inc)}, usable {len(ids)} (FM {nfm}, HC {nhc}), fc shape {fcarr.shape}")
    print(f"dropped: {drop if drop else 'none'}")
    return fcarr, labs, ids
def main():
    warnings.filterwarnings("ignore")
    fcarr, labs, ids=assemble()

    bl=baselines(fcarr, labs, preprocessed_fmri_dir=str(fmridir), subject_ids=ids)
    print(f"FM subset for pca/flat/ica: {len(bl.fc_matrices)} subjects, shape {bl.fc_matrices.shape}")
    print(f"full cohort for svm: {len(bl._fc_matrices_all)} subjects")

    rows=[]

    try:
        lab, s, X=bl.pca_kmeans(k)
        pp=_perm(X, lab); g=_guard(lab)
        rows.append({"baseline": "pca_kmeans", "metric": "silhouette", "value": float(s),
                     "k": k, "n_subjects": int(len(bl.fc_matrices)), "perm_p": pp, "passes_guard": g, "note": "FM only, mean-FC"})
        print(f"pca_kmeans k={k} sil={s:.6f} perm_p={pp:.4f} guard={g}")
    except Exception as e:
        rows.append({"baseline": "pca_kmeans", "metric": "silhouette", "value": np.nan,
                     "k": k, "n_subjects": int(len(bl.fc_matrices)), "perm_p": np.nan, "passes_guard": False, "note": f"failed: {type(e).__name__}: {e}"})
        print(f"pca_kmeans failed: {e}")

    try:
        lab, s, X=bl.flat_triangle_kmeans(k)
        pp=_perm(X, lab); g=_guard(lab)
        rows.append({"baseline": "flat_triangle_kmeans", "metric": "silhouette", "value": float(s),
                     "k": k, "n_subjects": int(len(bl.fc_matrices)), "perm_p": pp, "passes_guard": g, "note": "FM only, upper-triangle"})
        print(f"flat_triangle_kmeans k={k} sil={s:.6f} perm_p={pp:.4f} guard={g}")
    except Exception as e:
        rows.append({"baseline": "flat_triangle_kmeans", "metric": "silhouette", "value": np.nan,
                     "k": k, "n_subjects": int(len(bl.fc_matrices)), "perm_p": np.nan, "passes_guard": False, "note": f"failed: {type(e).__name__}: {e}"})
        print(f"flat_triangle_kmeans failed: {e}")

    try:
        acc=bl.svm_classification()
        flat=bl._fc_matrices_all.reshape(len(bl._fc_matrices_all), -1)
        pp=_acc_perm(flat, bl._labels_all, acc)
        rows.append({"baseline": "svm_classification", "metric": "cv_accuracy", "value": float(acc),
                     "k": np.nan, "n_subjects": int(len(bl._fc_matrices_all)), "perm_p": pp, "passes_guard": np.nan,
                     "note": f"FM vs HC, {config.svmCvFolds}-fold CV, label-perm p"})
        print(f"svm_classification {config.svmCvFolds}-fold cv acc={acc:.6f} perm_p={pp:.4f}")
    except Exception as e:
        rows.append({"baseline": "svm_classification", "metric": "cv_accuracy", "value": np.nan,
                     "k": np.nan, "n_subjects": int(len(bl._fc_matrices_all)), "perm_p": np.nan, "passes_guard": np.nan, "note": f"failed: {type(e).__name__}: {e}"})
        print(f"svm_classification failed: {e}")

    try:
        lab, s, bk, X=bl.group_ica_kmeans()
        pp=_perm(X, lab); g=_guard(lab)
        rows.append({"baseline": "group_ica_kmeans", "metric": "silhouette", "value": float(s),
                     "k": int(bk), "n_subjects": int(len(bl._fm_subject_ids)), "perm_p": pp, "passes_guard": g,
                     "note": "FM only, task-epr CanICA best-of-k in {2,3,4}"})
        print(f"group_ica_kmeans best_k={bk} sil={s:.6f} perm_p={pp:.4f} guard={g}")
    except Exception as e:
        rows.append({"baseline": "group_ica_kmeans", "metric": "silhouette", "value": np.nan,
                     "k": np.nan, "n_subjects": int(len(bl._fm_subject_ids)), "perm_p": np.nan, "passes_guard": False, "note": f"failed: {type(e).__name__}: {e}"})
        print(f"group_ica_kmeans failed: {e}")

    outdir=config.resultsRoot
    outdir.mkdir(parents=True, exist_ok=True)
    out=outdir/"baseline_comparison.csv"
    df=pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
