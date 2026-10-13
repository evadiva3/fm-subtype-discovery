import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from summarize_detection import summarize
import scope_detection as sd22
R=Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(d,replay=False):
    cfg=json.loads((d/"config.json").read_text(encoding="utf-8"))
    n=cfg["n"]
    for p,v in cfg["source_sha256"].items():
        assert sha(R/p)==v,p
    assert sha(sd22.PROTOCOL)==cfg["protocol_sha256"],"Protocol changed after execution"
    rs=[json.loads(l) for l in (d/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    keys={(r["spectrum"],r["q"],r["group"],r["index"]) for r in rs}
    assert len(keys)==len(rs)==sum(cfg["groups"].values())*4
    assert not any("error" in r for r in rs) and all(r["n"]==n for r in rs)
    assert len({r["reference_seed"] for r in rs}|{r["bootstrap_seed"] for r in rs})==2*len(rs)
    lookup={(r["spectrum"],r["q"],r["group"],r["index"]):r for r in rs}
    for r in rs:
        assert r["source_sha256"]==lookup[r["spectrum"],15-r["q"],r["group"],r["index"]]["source_sha256"],"Nested projections must share the source dataset"
        if r["group"]!="calibration":
            assert r["data_seed"]==lookup[r["spectrum"],r["q"],"null",r["index"]]["data_seed"],"Alternatives must share the evaluation noise"
            assert r["data_seed"]!=lookup[r["spectrum"],r["q"],"calibration",r["index"]]["data_seed"]
        if r["group"] not in ("calibration","null"):
            assert r["group_sizes"]==sd22.sizes(r["group"][:-1],n)
            if r["group"]!="shift4":
                assert abs(r["projected_mean_shift"])<1e-18,"Centered alternatives must have zero mixture mean"
        for a in ("W","D4","K","E"):
            z=r["p"][a]*(cfg["draws"]+1)
            assert np.isclose(z,round(z)) and 1<=round(z)<=cfg["draws"]+1
    assert summarize(d)==json.loads((d/"summary.json").read_text(encoding="utf-8")),"Summary mismatch"
    replayed=[]
    if replay:
        for key in [(0,5,"tetra4",0),(1,10,"pairuneq4",249),(0,10,"null",999)]:
            r=lookup[key]
            z=sd22.one((cfg["root_seed"],n,key[0],key[1],sd22.G.index(key[2]),key[3],cfg["draws"]))
            assert json.loads(json.dumps(z))==r,key
            replayed.append(list(key))
    out=dict(passed=True,n=n,records=len(rs),exact_replays=replayed,complete_ids=True,source_and_protocol_checksums=True,calibration_evaluation_streams_disjoint=True,alternatives_share_evaluation_noise=True,nested_projection_source_pairing=True,group_sizes=True,centered_alternatives_zero_mean=True,summary_recomputed=True)
    (d/"verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--directory",type=Path,nargs="+",default=[R/"results/revision_20260924/scope_detection_n28",R/"results/revision_20260924/scope_detection_n100"])
    p.add_argument("--replay",action="store_true")
    a=p.parse_args()
    for d in a.directory:
        audit(d,a.replay)
