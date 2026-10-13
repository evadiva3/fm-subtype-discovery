import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from summarize_detection import summarize
import asymmetric_detection as ad
R=Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(d,replay=False):
    cfg=json.loads((d/"config.json").read_text(encoding="utf-8"))
    for p,v in cfg["source_sha256"].items():
        assert sha(R/p)==v,p
    assert sha(ad.PROTOCOL)==cfg["protocol_sha256"],"Protocol changed after execution"
    rs=[json.loads(l) for l in (d/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    keys={(r["spectrum"],r["q"],r["group"],r["index"]) for r in rs}
    assert len(keys)==len(rs)==sum(cfg["groups"].values())*4
    assert not any("error" in r for r in rs)
    assert len({r["reference_seed"] for r in rs}|{r["bootstrap_seed"] for r in rs})==2*len(rs)
    lookup={(r["spectrum"],r["q"],r["group"],r["index"]):r for r in rs}
    for r in rs:
        assert r["source_sha256"]==lookup[r["spectrum"],15-r["q"],r["group"],r["index"]]["source_sha256"],"Nested projections must share the source dataset"
        if r["group"]!="calibration":
            assert r["data_seed"]==lookup[r["spectrum"],r["q"],"null",r["index"]]["data_seed"],"Alternatives must share the evaluation noise"
            assert r["data_seed"]!=lookup[r["spectrum"],r["q"],"calibration",r["index"]]["data_seed"]
        if r["group"].startswith("tri"):
            assert sorted(np.bincount(r["labels"]).tolist())==[9,9,10]
        elif r["group"][:-1] in ("antipodal","pair"):
            assert np.bincount(r["labels"]).tolist()==[14,14]
        for a in ("W","D4","K","E"):
            z=r["p"][a]*(cfg["draws"]+1)
            assert np.isclose(z,round(z)) and 1<=round(z)<=cfg["draws"]+1
    assert summarize(d)==json.loads((d/"summary.json").read_text(encoding="utf-8")),"Summary mismatch"
    replayed=[]
    if replay:
        for key in [(0,5,"tri4",0),(1,10,"antipodal4",249),(0,10,"null",999)]:
            r=lookup[key]
            z=ad.one((cfg["root_seed"],key[0],key[1],ad.G.index(key[2]),key[3],cfg["draws"]))
            assert json.loads(json.dumps(z))==r,key
            replayed.append(list(key))
    out=dict(passed=True,records=len(rs),exact_replays=replayed,complete_ids=True,source_and_protocol_checksums=True,calibration_evaluation_streams_disjoint=True,alternatives_share_evaluation_noise=True,nested_projection_source_pairing=True,balanced_labels=True,summary_recomputed=True)
    (d/"verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--directory",type=Path,default=R/"results/revision_20260924/asymmetric_detection")
    p.add_argument("--replay",action="store_true")
    a=p.parse_args()
    audit(a.directory,a.replay)
