import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v]="1"
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from headline_recorded import summary,init,one,ov,nc
from summarize_detection import summarize
R=Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    return json.loads(p.read_text(encoding="utf-8"))

def audit(d,replay=False):
    h=d/"headline"
    c=load(h/"config.json")
    rs=[json.loads(l) for l in (h/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    n=c["n_datasets"]
    assert len(rs)==n and {(r["block"],r["index"]) for r in rs}=={(b,i) for b in range(5) for i in range(n//5)}
    assert len({r["data_seed"] for r in rs}|{r["reference_seed"] for r in rs})==2*n
    assert len({r["data_sha256"] for r in rs})==n
    assert sha(R/"data/outputs/trained_fm_embeddings.npy")==c["input_sha256"]
    for p,v in c["source_sha256"].items():
        assert sha(R/p)==v,p
    for r in rs:
        for a in ("M","C"):
            if r["guard_failed"]:
                assert r["p"+a]==1.
                continue
            z=r[a]
            assert z["n_draws"]==c["n_draws"]
            assert r["p"+a]==z["p"]==(z["n_exceed"]+1)/(c["n_draws"]+1)
            assert sum(z["null_k_counts"].values())+z["n_degenerate"]==c["n_draws"]
        if not r["guard_failed"]:
            assert min(r["cluster_sizes"])>=nc.min_cluster_guard(c["embedding_shape"][0])
    hs=load(h/"summary.json")
    assert hs["summary"]==summary(rs)
    assert hs["records_sha256"]==sha(h/"records.jsonl")
    for b in range(5):
        assert hs["blocks"][str(b)]==summary([r for r in rs if r["block"]==b])
    replayed=[]
    if replay:
        x=nc.l2_normalize(np.load(R/"data/outputs/trained_fm_embeddings.npy").astype(np.float64))
        m,f=nc._gaussian_factor(x)
        init(m,f,len(x))
        for b,i in [(0,0),(4,n//5-1)]:
            r=next(r for r in rs if (r["block"],r["index"])==(b,i))
            z=one((b,i,int(r["data_seed"]),int(r["reference_seed"]),c["n_draws"]))
            assert json.loads(json.dumps(z))==r
            replayed.append([b,i])
    d2=d/"calibrated_detection"
    cfg=load(d2/"config.json")
    for p,v in cfg["source_sha256"].items():
        assert sha(R/p)==v,p
    assert sha(R/"docs/FOLLOWUP_PROTOCOL_20260924.md")==cfg["protocol_sha256"]
    cr=[json.loads(l) for l in (d2/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({r["reference_seed"] for r in cr}|{r["bootstrap_seed"] for r in cr})==2*len(cr)
    lookup={(r["spectrum"],r["q"],r["group"],r["index"]):r for r in cr}
    for r in cr:
        assert r["source_sha256"]==lookup[r["spectrum"],15-r["q"],r["group"],r["index"]]["source_sha256"]
        if r["group"]!="calibration":
            assert r["data_seed"]==lookup[r["spectrum"],r["q"],"null",r["index"]]["data_seed"]
            assert r["data_seed"]!=lookup[r["spectrum"],r["q"],"calibration",r["index"]]["data_seed"]
        for a in ("W","D4","K","E"):
            z=r["p"][a]*(cfg["draws"]+1)
            assert np.isclose(z,round(z)) and 1<=round(z)<=cfg["draws"]+1
    assert summarize(d2)==load(d2/"summary.json"),"Detection summary mismatch"
    out=dict(passed=True,headline_records=len(rs),detection_records=len(cr),headline_exact_replays=replayed,complete_ids=True,source_checksums=True,calibration_evaluation_streams_disjoint=True,nested_projection_source_pairing=True,all_saved_summaries_recomputed=True)
    (d/"verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--directory",type=Path,default=R/"results/revision_20260924")
    p.add_argument("--replay",action="store_true")
    a=p.parse_args()
    audit(a.directory,a.replay)
