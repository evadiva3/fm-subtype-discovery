#check stored evidence without rerunning anything
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(output):
    out={"scope":"Stored records and internal consistency; not a full simulation replay","angular":{},"paired":{},"source_hashes":{}}
    for study in ("angular_v9","angular_v9_balanced"):
        folder=ROOT/"results"/study
        summary=json.loads((folder/"summary.json").read_text())
        cfg=summary["config"]
        seen=set()
        total=0
        for key,cell in summary["cells"].items():
            recs=[json.loads((folder/key/f"{i:05}.json").read_text()) for i in range(cfg["datasets"])]
            assert all(("error" not in r for r in recs)),key
            for arm,summ in cell["arms"].items():
                events=sum((r["p"][arm]<=0.05 for r in recs))
                assert events==summ["count"],(key,arm)
                assert events/len(recs)==summ["rate"],(key,arm)
                ci95=binomtest(events,len(recs)).proportion_ci(method="exact")
                assert np.allclose([ci95.low,ci95.high],summ["ci"],atol=1e-10),(key,arm)
            seen.update((r["source_data_sha256"] for r in recs))
            total+=len(recs)
        out["angular"][study]={"records":total,"unique_sources":len(seen),"cells":len(summary["cells"]),"all_rates_and_intervals_match":True}
        for key,digest in cfg["source_sha256"].items():
            path=ROOT/("analysis" if key.startswith(("validate_","analyze_")) else "packaging")/key
            assert sha(path)==digest,(key,"source checksum mismatch")
            out["source_hashes"][path.relative_to(ROOT).as_posix()]=digest
    summ=json.loads((ROOT/"reproducibility/audits/audit_results.json").read_text())["paired_comparisons"]
    for name in ("reversal_verification","grid_verification","statistic_verification"):
        res=json.loads((ROOT/"results"/(name+".json")).read_text())
        for key,cell in res["cells"].items():
            label="grid_"+key if name=="grid_verification" else {"nwb":"negative_within_between","ndb":"negative_Davies_Bouldin"}.get(key,key)
            disc=cell["discordant"]
            conly=sum((r["pC"]<0.05<=r["pM"] for r in disc))
            monly=sum((r["pM"]<0.05<=r["pC"] for r in disc))
            assert (conly,monly)==(cell["C_only"],cell["M_only"]),(name,key)
            assert len({r["seed"] for r in disc})==len(disc)
            assert cell["n_datasets"]==1600
            assert (conly,monly)==(summ[label]["c_only"],summ[label]["m_only"]),label
            pval=binomtest(conly,conly+monly,0.5).pvalue if conly+monly else 1.0
            assert np.isclose(pval,summ[label]["exact_two_sided_McNemar_p"],rtol=1e-10,atol=1e-300)
            out["paired"][label]={"C_only":conly,"M_only":monly,"discordant_records":len(disc),"verified":True}
    head=json.loads((ROOT/"results/headline_embedding.json").read_text())
    for arm in ("M","C"):
        assert sum((r[arm+"_events"] for r in head["blocks"]))==head[arm]["events"]
        assert head[arm]["events"]/head["n_datasets"]==head[arm]["rejection_rate"]
    assert sum(head["selected_k_counts"].values())==head["n_datasets"]
    sys.path.insert(0,str(ROOT/"experiments/ds002748_mdd/propagated_null"))
    import nullcal
    emb=np.load(ROOT/head["embedding"],allow_pickle=False)
    a=nullcal.l2_normalize(emb)
    fac,b=nullcal._gaussian_factor(a)
    out["headline"]={"array_shape":list(emb.shape),"embedding_sha256":sha(ROOT/head["embedding"]),"fitted_mean_norm":float(np.linalg.norm(fac)),"nonzero_mean":bool(np.linalg.norm(fac)>1e-10),"covariance_factor_shape":list(b.shape),"block_totals_match":True,"limitation":"Only aggregate and block results were retained for the historical 10000-dataset headline run."}
    out["passed"]=True
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/revision_20260923/repository_audit.json")
    audit(parser.parse_args().output)
