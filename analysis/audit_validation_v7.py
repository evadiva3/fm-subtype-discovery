#Verify saved v7 records, source provenance, summaries and optional replay.
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"analysis"))
import validate_cluster_nulls as study
nv=study.nv

def audit(path,replay=False):
    final=json.loads((path/"summary.json").read_text())
    cfg=json.loads((path/"config.json").read_text())
    assert cfg==final["config"]
    for name,digest in cfg["source_sha256"].items():
        p=ROOT/("analysis" if name.startswith("validate_") else "packaging")/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,("source changed",p)
    specs={c["name"]:c for c in json.loads((path/"populations.json").read_text())}
    all_seeds=[];all_hashes=[];records={};replayed=[]
    for name in cfg["cells"]:
        cell=specs[name]
        rows=[]
        for i in range(cfg["datasets"]):
            r=json.loads((path/name/f"{i:05d}.json").read_text())
            rep,idx=divmod(i,cfg["block_size"])
            expected=nv.stream_seed(cfg["root_seed"],7,cell["id"],rep,idx,0)
            assert r["index"]==i and int(r["data_seed"])==expected
            assert r["replicate"]==rep and r["local_index"]==idx
            assert r["seed"]==str(nv.stream_seed(cfg["root_seed"],7,cell["id"],rep,idx,1))
            x=nv.normalize(np.random.default_rng(expected).standard_normal((cell["n"],cell["d"]))@np.asarray(cell["population_factor"]))
            assert hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()==r["data_sha256"]
            assert all(np.isfinite(v) and 0<v<=1 for v in r["p"].values())
            assert all(abs(v*(cfg["draws"]+1)-round(v*(cfg["draws"]+1)))<1e-10 for v in r["p"].values())
            assert r["p"]["S"]>=r["p"]["Mg"] and r["p"]["C"]>=r["p"]["Gg"]
            rows.append(r);all_seeds.append(r["data_seed"]);all_hashes.append(r["data_sha256"])
        summary=study.summarize(rows)
        assert summary==final["cells"][name]
        assert summary==json.loads((path/name/"summary.json").read_text())
        if replay:
            assert study.one((cell,0,cfg))==rows[0],("replay failed",name)
            replayed.append(name)
        records[name]=rows
    assert len(all_seeds)==len(set(all_seeds))==final["total_unique_datasets"]
    assert len(all_hashes)==len(set(all_hashes))
    return dict(records=len(all_seeds),unique_datasets=len(set(all_hashes)),source_hashes_verified=True,
                regenerated_dataset_hashes=len(all_hashes),summaries_recomputed=True,replayed_cells=replayed),records

def main():
    p=argparse.ArgumentParser();p.add_argument("--replay",action="store_true")
    args=p.parse_args()
    primary,rows=audit(ROOT/"results/validation_v7",args.replay)
    resolution,high=audit(ROOT/"results/validation_v7_resolution",args.replay)
    paired={}
    for name,rr in high.items():
        low=rows[name][:len(rr)]
        assert [r["data_sha256"] for r in low]==[r["data_sha256"] for r in rr]
        paired[name]=dict(low=study.summarize(low),high=study.summarize(rr),decision_changes={
            arm:sum((a["p"][arm]<=.05)!=(b["p"][arm]<=.05) for a,b in zip(low,rr)) for arm in rr[0]["p"]})
    result=dict(primary=primary,resolution=resolution,paired_resolution=paired)
    out=ROOT/"results/validation_v7/audit.json";out.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(primary=primary,resolution=resolution),indent=2))

if __name__=="__main__":main()
