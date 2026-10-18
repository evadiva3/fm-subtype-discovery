#recompute the S9 cells from their records
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/ds002748_mdd/propagated_null"))
import nullcal as nc

def verify(path):
    saved=json.loads(path.read_text());rows=saved["records"];cfg=saved["config"]
    assert cfg["datasets"]==120 and cfg["draws"]==200 and len(rows)==1560
    assert all("error" not in r for r in rows)
    seeds=[r[k] for r in rows for k in ("data_seed","draw_seed")]
    assert len(set(seeds))==len(seeds)
    assert len({r["source_sha256"] for r in rows})==1560
    for name,expected in cfg["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for r in rows:
        assert all(0<r[k]<=1 and abs(r[k]*201-round(r[k]*201))<1e-9 for k in ("pM","pC"))
        if r["guard_failed"]:assert r["pM"]==r["pC"]==1.
    for key,c in saved["cells"].items():
        rr=[r for r in rows if r["cell"]==int(key)]
        assert len(rr)==120
        assert np.isclose(c["pr_mean"],np.mean([r["pr"] for r in rr]))
        for arm in ("M","C"):
            k=sum(r["p"+arm]<.05 for r in rr)
            assert k==c[arm]["events"] and k/120==c[arm]["rate"]
            assert np.allclose(c[arm]["wilson95"],nc._wilson(k,120))
        assert sum(r["pM"]<.05<=r["pC"] for r in rr)==c["M_only"]
        assert sum(r["pC"]<.05<=r["pM"] for r in rr)==c["C_only"]
    report=dict(passed=True,records=len(rows),distinct_sources=1560,distinct_seed_addresses=len(seeds),
        numerical_failures=0,guard_failures=sum(r["guard_failed"] for r in rows),
        M_only=sum(c["M_only"] for c in saved["cells"].values()),C_only=sum(c["C_only"] for c in saved["cells"].values()),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_name("geometry_audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"results/revision_20260923/geometry_surface_fresh.json")
    verify(p.parse_args().input)
