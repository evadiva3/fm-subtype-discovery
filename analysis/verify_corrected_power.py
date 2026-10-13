#check the corrected eigenvector study records and summaries
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]
STUDIES={"shape_aware_article_geometry":("shape_aware_at_article_geometry",1200),"shape_aware_detection_floor":("shape_aware_detection_floor",2800),"shape_aware_minor_axis":("shape_aware_minor_axis",1200)}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(folder):
    out={"studies":{},"scope":"Complete corrected runs, 200 datasets per cell and 199 reference draws"}
    src_hashes,seeds=(set(),set())
    for name,(mod,nrec) in STUDIES.items():
        path=folder/(name+"_eigenvector_corrected.json")
        if not path.exists() and name!=mod:
            path=folder/(mod+"_eigenvector_corrected.json")
        res=json.loads(path.read_text())
        assert res["n_datasets"]==200 and res["draws"]==199
        assert res["source_sha256"]==sha(ROOT/"analysis"/(mod+".py")),mod
        assert res["spectrum_file_sha256"]==sha(ROOT/"results/overnight/e2_spectrum_shape.json")
        for modname,val in res["module_sha256"].items():
            p=ROOT/("experiments/ds002748_mdd/propagated_null" if modname=="ov_common" else "packaging")/(modname+".py")
            assert sha(p)==val,modname
        recs=res["per_dataset_results"]
        assert len(recs)==nrec and all(("error" not in r for r in recs))
        assert len({(r["cell"],r["q"],r["index"]) for r in recs})==nrec
        hashes={r["source_data_sha256"] for r in recs}
        assert len(hashes)==nrec and (not hashes&src_hashes)
        src_hashes|=hashes
        seed_list=[r[k] for r in recs for k in ("source_data_seed","null_seed","bootstrap_seed")]
        assert len(set(seed_list))==len(seed_list) and (not set(seed_list)&seeds)
        seeds.update(seed_list)
        for rec in recs:
            assert all((0<p<=1 and abs(p*200-round(p*200))<1e-09 for p in rec["p"].values()))
            assert rec["diagnostics"]["shape_fit"]["whitened_moment_residual"]<1e-07
            assert rec["diagnostics"]["max_reference_residual"]<1e-07
        for key,cell in res["results"].items():
            rows=[r for r in recs if r["cell"]==cell["cell"] and r["q"]==cell["q"]]
            assert len(rows)==200
            for arm,val in cell["arms"].items():
                events=sum((r["p"][arm]<=0.05 for r in rows))
                assert events==val["count"] and events/200==val["rate"]
                ci95=binomtest(events,200).proportion_ci(method="exact")
                assert np.allclose([ci95.low,ci95.high],val["ci"],atol=1e-10)
        out["studies"][name]={"records":nrec,"cells":len(res["results"]),"verified":True,"guard_failures":sum((r["diagnostics"]["guard_failed"] for r in recs)),"numerical_failures":0,"result_sha256":sha(path)}
    out.update(passed=True,records=len(src_hashes),distinct_source_datasets=len(src_hashes),distinct_seed_addresses=len(seeds))
    (folder/"power_audit.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,default=ROOT/"results/revision_20260923")
    verify(p.parse_args().directory)
