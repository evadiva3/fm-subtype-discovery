import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import robustness_detection as rd
R=Path(__file__).resolve().parents[1]
D=R/"results/revision_20260925"

def rows(d):
    return [json.loads(l) for l in (d/"records.jsonl").read_text(encoding="utf-8").splitlines()]

def checksums(cfg):
    for p,v in cfg["source_sha256"].items():
        assert rd.sha(R/p)==v,p
    assert rd.sha(rd.PROTOCOL)==cfg["protocol_sha256"],"Protocol changed after execution"

def audit_a(d,replay):
    cfg=json.loads((d/"config.json").read_text(encoding="utf-8"));n=cfg["n"]
    checksums(cfg)
    rs=rows(d)
    lookup={(r["spectrum"],r["q"],r["group"],r["index"]):r for r in rs}
    assert len(lookup)==len(rs)==sum(cfg["groups"].values())*4
    assert not any("error" in r for r in rs) and all(r["n"]==n for r in rs)
    assert len({r["reference_seed"] for r in rs}|{r["bootstrap_seed"] for r in rs})==2*len(rs)
    seeds={g:{r["data_seed"] for r in rs if r["group"]==g} for g in cfg["groups"]}
    for a,b in (("calibration","null"),("calibration","skew"),("calibration","heavy"),("null","skew"),("null","heavy"),("skew","heavy")):
        assert not seeds[a]&seeds[b],(a,b)
    for r in rs:
        assert r["source_sha256"]==lookup[r["spectrum"],15-r["q"],r["group"],r["index"]]["source_sha256"],"Nested projections must share the source dataset"
        if r["group"] in ("rand34","hexa4"):
            assert r["data_seed"]==lookup[r["spectrum"],r["q"],"null",r["index"]]["data_seed"],"Alternatives must share the evaluation noise"
            assert r["group_sizes"]==rd.sizes(r["group"],n)
            assert abs(r["projected_mean_shift"])<1e-18,"Centered alternatives must have zero mixture mean"
        for a in ("W","D4","K","E"):
            z=r["p"][a]*(cfg["draws"]+1)
            assert np.isclose(z,round(z)) and 1<=round(z)<=cfg["draws"]+1
    assert rd.summarize(d)==json.loads((d/"summary.json").read_text(encoding="utf-8")),"Summary mismatch"
    replayed=[]
    if replay:
        for key in [(0,5,"rand34",0),(1,10,"hexa4",249),(0,10,"skew",999),(1,5,"heavy",500)]:
            z=rd.one_a((cfg["root_seed"],n,key[0],key[1],rd.GA.index(key[2]),key[3],cfg["draws"]))
            assert json.loads(json.dumps(z))==lookup[key],key
            replayed.append(list(key))
    return dict(passed=True,study="S29a",n=n,records=len(rs),exact_replays=replayed,streams_disjoint=True,alternatives_share_evaluation_noise=True,nested_projection_source_pairing=True,centered_alternatives_zero_mean=True,summary_recomputed=True)

def gaussian_path():
    # with Gaussian coordinates the S29b draw is the S24 draw: an S24 record's source hash must be reproduced
    s24=rd.s24
    r=next(json.loads(l) for l in (D/"mc_calibrated_sensitivity/records.jsonl").read_text(encoding="utf-8").splitlines() if '"group": "null"' in l)
    rng=np.random.default_rng(int(r["data_seed"]))
    x=rd.nc.l2_normalize(s24.MEAN+rd.noise("gauss",rng,(s24.N,s24.FACTOR.shape[0]))@s24.FACTOR)
    assert hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()==r["source_sha256"],"S29b draw differs from S24"
    return r["index"]

def audit_b(d,replay):
    cfg=json.loads((d/"config.json").read_text(encoding="utf-8"))
    checksums(cfg)
    rs=rows(d)
    lookup={(r["group"],r["index"]):r for r in rs}
    assert len(lookup)==len(rs)==sum(cfg["groups"].values()) and not any("error" in r for r in rs)
    assert len({r["data_seed"] for r in rs})==len({r["reference_seed"] for r in rs})==len(rs)
    for r in rs:
        for a in ("M","C"):
            z=r["p"][a]*(cfg["draws"]+1)
            assert r["guard_failed"] or (np.isclose(z,round(z)) and 1<=round(z)<=cfg["draws"]+1)
    assert rd.summarize_b(d)==json.loads((d/"summary.json").read_text(encoding="utf-8")),"Summary mismatch"
    replayed=[]
    if replay:
        for key in [("skew",0),("heavy",999)]:
            z=rd.one_b((cfg["root_seed"],rd.GB.index(key[0]),key[1],cfg["draws"]))
            assert json.loads(json.dumps(z))==lookup[key],key
            replayed.append(list(key))
    return dict(passed=True,study="S29b",records=len(rs),exact_replays=replayed,gaussian_path_reproduces_s24_record=gaussian_path(),summary_recomputed=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--replay",action="store_true")
    a=p.parse_args()
    out=[audit_b(D/"robustness_mc",a.replay)]+[audit_a(D/f"robustness_n{n}",a.replay) for n in (28,100)]
    (D/"robustness_verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
