import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower
import os
import sys
from pathlib import Path

_ROOT=Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src", _ROOT / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import config


class clinical_validator:
    def __init__(self,clinical_csv_path=None):
        clinical_csv_path=config.clinicalCsv if clinical_csv_path is None else clinical_csv_path
        self.df=pd.read_csv(clinical_csv_path)
        self.count_vars=['age', 'vas_pain', 'hamd_total', 'hama_total', 'tas_total', 
        'tas_dif', 'tas_ddf', 'tas_eot', 'erq_reappraisal', 'erq_suppression']


    def _label_df(self, labels_path=None):
        labels_path=config.clusterOutput / "K-Means-Labeling.csv" if labels_path is None else Path(labels_path)
        lab=pd.read_csv(labels_path)
        lab=pd.DataFrame({"subject_id": lab["Subject_Id"].astype(str), "temp": lab["Label"]})
        df=self.df.copy()
        df["subject_id"]=df["subject_id"].astype(str)
        return df.merge(lab, on="subject_id", how="inner")

    def compare_groups(self, subtype=None, labels_path=None):
        merged=self._label_df(labels_path)
        labs=sorted(merged['temp'].unique())
        if len(labs)<2:
            raise ValueError(f"compare_groups needs 2 subtypes, got {labs}")
        if len(labs)>2:
            print(f"[clinical_validation] WARNING: {len(labs)} subtypes present {labs}; comparing only {labs[0]} vs {labs[1]}, rest dropped")
        group=merged[merged['temp']==labs[0]]
        group1=merged[merged['temp']==labs[1]]
        results=[]
        for i in self.count_vars:
            stat, p=stats.mannwhitneyu(group[i], group1[i], alternative='two-sided')
            #above is man-whitney u math
            mean0=group[i].mean()
            mean1=group1[i].mean()
            std0=group[i].std()
            std1=group1[i].std()
            pooled_std = np.sqrt((std0**2 + std1**2) / 2)
            d=(mean0 - mean1)/pooled_std
            results.append({
                'variable': i,
                'p_value': p,
                'cohens_d': d
                            })
            #above is cohen
        results_df=pd.DataFrame(results)
        _,corrected_p, _, _=multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_corrected']=corrected_p
        results_df['significant']=corrected_p < config.fdrAlpha
        return results_df
    
    def compute_effect_sizes(self,subtype=None,labels_path=None):
        # subtype arg kept for API compat, labels now joined by subject_id not position
        merged=self._label_df(labels_path)
        group=merged[merged['temp']==0]
        group1=merged[merged['temp']==1]
        effect_sizes={}
        for i in self.count_vars:
            mean=group[i].mean()
            mean1=group1[i].mean()
            std=group[i].std()
            std1=group1[i].std()
            pooled_std=np.sqrt((std**2 + std1**2) / 2)
            effect_sizes[i]=(mean - mean1)/pooled_std
        return effect_sizes



    def project_sample_size(self,results_df,target_power=0.80,alpha=0.05):
        # given each variables already computed Cohen's d, this estimates the n per group a
        # follow up study would need to reach target_power. This plans a future study it does
        # not judge whether the current results are significant.
        # statsmodels has no direct nonparametric (Mann-Whitney U) power solver, so TTestIndPower
        # is used as the closest parametric approximation of the required sample size
        solver=TTestIndPower()
        projected_n=[]
        for d in results_df['cohens_d']:
            effect=abs(d)
            if effect==0 or np.isnan(effect):
                projected_n.append(np.nan)
                continue
            n=solver.solve_power(effect_size=effect,alpha=alpha,power=target_power,ratio=1.0,alternative='two-sided')
            projected_n.append(np.ceil(n))
        results_df=results_df.copy()
        results_df['required_n_for_80_power']=projected_n
        return results_df


    def run_all(self,subtype,save_dir=None):
        save_dir=config.resultsRoot if save_dir is None else save_dir
        os.makedirs(save_dir, exist_ok=True)
        df=self.compare_groups(subtype)
        df=self.project_sample_size(df)
        effect_sizes=self.compute_effect_sizes(subtype)
        df.to_csv(os.path.join(save_dir, 'clinical_validation_results.csv'), index=False)
        print(effect_sizes)
        return df


#per-subtype FD (motion) check

def _load_labels(path=None):
    path=config.clusterOutput/"K-Means-Labeling.csv" if path is None else Path(path)
    if not path.exists():
        return None, f"missing subtype labels ({path})"
    df=pd.read_csv(path)
    return pd.DataFrame({"sid": df["Subject_Id"], "lab": df["Label"]}), None

def _load_fd(path=None):
    path=config.exclusionManifestPath if path is None else Path(path)
    if not path.exists():
        return None, f"missing FD manifest ({path})"
    df=pd.read_csv(path)
    cols=[c for c in df.columns if c.startswith("meanFD_")]
    if not cols:
        return None, f"no meanFD_* columns in {path}"
    if "exclusion_reason" in df.columns:
        df=df[df["exclusion_reason"].fillna("").astype(str)==""] 
    return pd.DataFrame({"sid": df["subject_id"], "fd": df[cols].mean(axis=1)}), None

#MannWhitney FD across 2 subtypes
def subtype_fd_test(lab, fd):
    m=lab.merge(fd, on="sid")
    if len(m)==0:
        return None, "no overlap between labeled subjects and FD subjects"
    grps=sorted(m["lab"].unique())
    if len(grps)!=2:
        return None, f"MannWhitney needs 2 subtypes, got {len(grps)}"
    a=m[m["lab"]==grps[0]]["fd"].dropna()
    b=m[m["lab"]==grps[1]]["fd"].dropna()
    u,p=stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(u), "p": float(p), "n0": int(len(a)), "n1": int(len(b))}, None

def run_fd_comparison(labels_path=None, fd_path=None):
    lab, blk=_load_labels(labels_path)
    if lab is None:
        return None, blk
    fd, blk=_load_fd(fd_path)
    if fd is None:
        return None, blk
    return subtype_fd_test(lab, fd)


def structural_self_test():
    #two fake subtypes
    rng=np.random.default_rng(config.randomSeed)
    n=20
    lab=pd.DataFrame({"sid": [f"s{i}" for i in range(2*n)], "lab": [0]*n+[1]*n})

    print("[self-test] FD differs between subtypes -> significant p ...")
    fd=pd.DataFrame({"sid": [f"s{i}" for i in range(2*n)], "fd": np.concatenate([rng.normal(0.10, 0.02, n), rng.normal(0.30, 0.02, n)])})
    res, blk=subtype_fd_test(lab, fd)
    assert blk is None, blk
    print(f"  U={res['U']:.1f} p={res['p']:.2e}")
    assert res["p"]<0.05, "real FD diff must be significant"

    print("[self-test] no FD difference -> non-significant p ...")
    fd2=pd.DataFrame({"sid": [f"s{i}" for i in range(2*n)], "fd": rng.normal(0.20, 0.05, 2*n)})
    res2, blk2=subtype_fd_test(lab, fd2)
    assert blk2 is None, blk2
    print(f"  U={res2['U']:.1f} p={res2['p']:.4f}")
    assert res2["p"]>0.05, "no FD diff must be non-significant"
    print("[self-test] Good! FD subtype test behaves.")
    return True


def main():
    os.makedirs(config.resultsRoot, exist_ok=True)
    print("=" * 70)
    print("Per-subtype FD motion check (paper Section 4.1)")
    print("=" * 70)
    res, blk=run_fd_comparison()
    if res is not None:
        out=config.resultsRoot / "subtype_fd_comparison.csv"
        pd.DataFrame([res]).to_csv(out, index=False)
        print(f"[fd] wrote {out}")
        print(f"  U={res['U']:.2f} p={res['p']:.4f} n0={res['n0']} n1={res['n1']}")
        if res["p"]>=config.fdrAlpha:
            print("  non-significant -> subtypes NOT explained by motion (desired)")
        else:
            print("  SIGNIFICANT -> motion may confound subtypes (flag)")
    else:
        print("-" * 70)
        print(f"FD comparison BLOCKED: {blk}")
        print("-" * 70)
        print("Running structural self-test on synthetic FD instead ...")
        structural_self_test()


if __name__ == "__main__":
    main()
    

        





