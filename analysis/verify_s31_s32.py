import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import argparse,json
from pathlib import Path
import recorded_controls as rc
R=Path(__file__).resolve().parents[1]
D=R/"results/revision_20260926"

def rows(d):
    return [json.loads(l) for l in (d/"records.jsonl").read_text(encoding="utf-8").splitlines()]

def checksums(cfg):
    for p,v in cfg["source_sha256"].items():
        assert rc.sha(R/p)==v,p
    for p,v in cfg["input_sha256"].items():
        assert rc.sha(R/p)==v,p
    assert rc.sha(rc.PROTOCOL)==cfg["protocol_sha256"],"Protocol changed after execution"

def seeds_distinct(rs):
    s=[r["data_seed"] for r in rs]+[r["reference_seed"] for r in rs]+[r["retest"]["reference_seed"] for r in rs if r.get("retest")]
    assert len(s)==len(set(s)),"Seed streams overlap"

def audit(d,study,replay):
    cfg=json.loads((d/"config.json").read_text(encoding="utf-8"))
    checksums(cfg)
    rs=rows(d)
    seeds_distinct(rs)
    fn=rc.summarize_a if study=="S31" else rc.summarize_b
    assert json.loads(json.dumps(fn(d)))==json.loads((d/"summary.json").read_text(encoding="utf-8")),"Summary mismatch"
    for r in rs:
        for v in r["p"].values():
            z=v*(cfg["draws"]+1)
            assert r["guard_failed"] or abs(z-round(z))<1e-9
    replayed=[]
    if replay:
        rc.init(rc.spectra(cfg["root_seed"]))
        if study=="S31":
            keys=[("cell1",0),("cell3",199),("cell6",399)]
            look={(r["cell"],r["index"]):r for r in rs}
            for c,i in keys:
                assert json.loads(json.dumps(rc.one_a((cfg["root_seed"],c,i,cfg["draws"]))))==look[c,i],(c,i)
                replayed.append([c,i])
        else:
            look={(r["block"],r["index"]):r for r in rs}
            disc=next(k for k,r in look.items() if r["retest"])
            for b,i in [(0,0),(3,399),disc]:
                assert json.loads(json.dumps(rc.one_b((cfg["root_seed"],b,i,cfg["draws"],cfg["retest_draws"]))))==look[b,i],(b,i)
                replayed.append([b,i])
    return dict(passed=True,study=study,records=len(rs),exact_replays=replayed,checksums=True,streams_distinct=True,summary_recomputed=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--replay",action="store_true")
    a=p.parse_args()
    out=[audit(D/"mechanism_control","S31",a.replay),audit(D/"direction_reversal","S32",a.replay)]
    (D/"verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
