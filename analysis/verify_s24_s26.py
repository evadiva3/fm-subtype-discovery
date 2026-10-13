#check the S24, S25 and S26 records
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from summarize_detection import summarize
R=Path(__file__).resolve().parents[1]
D=R/"results/revision_20260925"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def records(d):
    return [json.loads(l) for l in (d/"records.jsonl").read_text(encoding="utf-8").splitlines()]

def common(d,cfg,rows,protocol):
    for p,v in cfg["source_sha256"].items():
        assert sha(R/p)==v,p
    assert sha(protocol)==cfg["protocol_sha256"],"Protocol changed after execution"
    assert not any("error" in r for r in rows)

def s24(d,replay):
    import mc_calibrated_sensitivity as m
    cfg=load(d/"config.json");rows=records(d)
    common(d,cfg,rows,m.PROTOCOL)
    assert sha(m.EMB)==cfg["input_sha256"]
    keys={(r["group"],r["index"]) for r in rows}
    assert len(keys)==len(rows)==sum(cfg["groups"].values())
    look={(r["group"],r["index"]):r for r in rows}
    for r in rows:
        if r["group"]!="calibration":
            assert r["data_seed"]==look["null",r["index"]]["data_seed"]!=look["calibration",r["index"]]["data_seed"]
        for a in ("M","C"):
            z=r["p"][a]*(cfg["draws"]+1)
            assert r["guard_failed"] or (np.isclose(z,round(z)) and 1<=round(z)<=cfg["draws"]+1)
    assert summarize(d)==load(d/"summary.json"),"S24 summary mismatch"
    rep=[]
    if replay:
        for g,i in (("two1.0",0),("three2.0",249),("null",999)):
            z=m.one((cfg["root_seed"],m.G.index(g),i,cfg["draws"]))
            assert json.loads(json.dumps(z))==look[g,i],(g,i);rep.append([g,i])
    return dict(records=len(rows),exact_replays=rep)

def s25(d,replay):
    import sample_size_detection as s
    cfg=load(d/"config.json");rows=records(d);n=cfg["n"]
    common(d,cfg,rows,s.PROTOCOL)
    keys={(r["spectrum"],r["q"],r["group"],r["index"]) for r in rows}
    assert len(keys)==len(rows)==sum(cfg["groups"].values())*4 and all(r["n"]==n for r in rows)
    look={(r["spectrum"],r["q"],r["group"],r["index"]):r for r in rows}
    for r in rows:
        assert r["source_sha256"]==look[r["spectrum"],15-r["q"],r["group"],r["index"]]["source_sha256"]
        if r["group"]!="calibration":
            assert r["data_seed"]==look[r["spectrum"],r["q"],"null",r["index"]]["data_seed"]
        if r["group"] not in ("calibration","null"):
            assert r["group_sizes"]==s.sizes(r["group"][:-1],n)
            if r["group"]!="shift4":
                assert abs(r["projected_mean_shift"])<1e-18
    assert summarize(d)==load(d/"summary.json"),"S25 summary mismatch"
    rep=[]
    if replay:
        for key in ((0,5,"tetra4",0),(1,10,"null",999)):
            z=s.base.one((cfg["root_seed"],n,key[0],key[1],s.base.G.index(key[2]),key[3],199))
            assert json.loads(json.dumps(z))==look[key],key;rep.append(list(key))
    return dict(n=n,records=len(rows),exact_replays=rep)

def s26(d,replay):
    import grid_recorded as g
    cfg=load(d/"config.json");rows=records(d);sm=load(d/"summary.json")
    common(d,cfg,rows,g.PROTOCOL)
    assert sha(g.HIST)==cfg["tau_source_sha256"]
    assert {(r["cell"],r["index"]) for r in rows}=={(c,i) for c in range(len(g.CELLS)) for i in range(cfg["datasets_per_cell"])}
    assert len({r["data_seed"] for r in rows}|{r["reference_seed"] for r in rows})==2*len(rows)
    assert g.summary(rows)==sm["cells"] and sm["records_sha256"]==sha(d/"records.jsonl")
    rep=[]
    if replay:
        look={(r["cell"],r["index"]):r for r in rows}
        for key in ((0,0),(len(g.CELLS)-1,cfg["datasets_per_cell"]-1)):
            z=g.one((cfg["root_seed"],key[0],key[1],cfg["draws"]))
            assert json.loads(json.dumps(z))==look[key],key;rep.append(list(key))
    return dict(records=len(rows),exact_replays=rep)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--replay",action="store_true")
    a=a.parse_args()
    out=dict(S24=s24(D/"mc_calibrated_sensitivity",a.replay),S25_n50=s25(D/"sample_size_n50",a.replay),S25_n200=s25(D/"sample_size_n200",a.replay),S26=s26(D/"grid_recorded",a.replay),passed=True)
    (D/"verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
