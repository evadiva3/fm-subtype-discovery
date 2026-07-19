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
            raise ValueError(f"compare_groups needs >=2 subtypes, got {labs}")
        groups={g: merged[merged['temp']==g] for g in labs}
        n_groups=len(labs)
        results=[]
        for i in self.count_vars:
            samples=[groups[g][i].dropna() for g in labs]
            n_total=int(sum(len(s) for s in samples))
            stat, p=stats.kruskal(*samples)
            eps_sq=(stat-n_groups+1)/(n_total-n_groups) if n_total>n_groups else np.nan
            row={'variable': i, 'H_stat': float(stat), 'p_value': float(p),
                 'epsilon_sq': float(eps_sq), 'n_groups': n_groups}
            for g in labs:
                row[f'mean_{g}']=groups[g][i].mean()
            if n_groups==2:
                m0,m1=groups[labs[0]][i].mean(),groups[labs[1]][i].mean()
                s0,s1=groups[labs[0]][i].std(),groups[labs[1]][i].std()
                row['cohens_d']=(m0-m1)/np.sqrt((s0**2+s1**2)/2)
            else:
                row['cohens_d']=np.nan
            results.append(row)
        results_df=pd.DataFrame(results)
        _,corrected_p, _, _=multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_corrected']=corrected_p
        results_df['significant']=corrected_p < config.fdrAlpha
        return results_df
    
    def compute_effect_sizes(self,subtype=None,labels_path=None):
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

def subtype_fd_test(lab, fd):
    m=lab.merge(fd, on="sid")
    if len(m)==0:
        return None, "no overlap between labeled subjects and FD subjects"
    grps=sorted(m["lab"].unique())
    if len(grps)<2:
        return None, f"kruskal needs >=2 subtypes, got {len(grps)}"
    samps=[m[m["lab"]==g]["fd"].dropna() for g in grps]
    h,p=stats.kruskal(*samps)
    ns={f"n_{g}": int(len(s)) for g,s in zip(grps, samps)}
    return {"H": float(h), "p": float(p), "k": len(grps), **ns}, None

def run_fd_comparison(labels_path=None, fd_path=None):
    lab, blk=_load_labels(labels_path)
    if lab is None:
        return None, blk
    fd, blk=_load_fd(fd_path)
    if fd is None:
        return None, blk
    return subtype_fd_test(lab, fd)


def structural_self_test():
    rng=np.random.default_rng(config.randomSeed)
    n=20
    lab=pd.DataFrame({"sid": [f"s{i}" for i in range(2*n)], "lab": [0]*n+[1]*n})

    print("self-test: FD differs, want significant")
    fd=pd.DataFrame({"sid": [f"s{i}" for i in range(2*n)], "fd": np.concatenate([rng.normal(0.10, 0.02, n), rng.normal(0.30, 0.02, n)])})
    res, blk=subtype_fd_test(lab, fd)
    assert blk is None, blk
    print(f"H={res['H']:.1f} p={res['p']:.2e}")
    assert res["p"]<0.05, "real FD diff must be significant"

    print("self-test: no FD diff, want non-significant")
    fd2=pd.DataFrame({"sid": [f"s{i}" for i in range(2*n)], "fd": rng.normal(0.20, 0.05, 2*n)})
    res2, blk2=subtype_fd_test(lab, fd2)
    assert blk2 is None, blk2
    print(f"H={res2['H']:.1f} p={res2['p']:.4f}")
    assert res2["p"]>0.05, "no FD diff must be non-significant"
    print("self-test ok")
    return True


def main():
    os.makedirs(config.resultsRoot, exist_ok=True)
    print("clinical comparison across subtypes (10 variables)")
    cv=clinical_validator()
    tab=cv.run_all(None)
    print(tab.to_string(index=False))
    print(f"wrote {config.resultsRoot / 'clinical_validation_results.csv'}")

    print("per-subtype FD motion check")
    res, blk=run_fd_comparison()
    if res is not None:
        out=config.resultsRoot / "subtype_fd_comparison.csv"
        pd.DataFrame([res]).to_csv(out, index=False)
        print(f"wrote {out}")
        ncols=" ".join(f"{k}={v}" for k, v in res.items() if k.startswith("n_"))
        print(f"H={res['H']:.2f} p={res['p']:.4f} k={res['k']} {ncols}")
        if res["p"]>=config.fdrAlpha:
            print("non-significant, subtypes not explained by motion")
        else:
            print("significant, motion may confound subtypes")
    else:
        print(f"FD comparison blocked: {blk}")
        print("running self-test on synthetic FD instead")
        structural_self_test()


if __name__ == "__main__":
    main()
    

        





