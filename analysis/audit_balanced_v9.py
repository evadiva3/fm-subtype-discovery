#Audit every balanced-kernel follow-up record and selected complete replays.
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[name]="1"
import argparse,hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import validate_balanced_v9 as follow
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,default=ROOT);p.add_argument("--replay",action="store_true");p.add_argument("--workers",type=int,default=8)
    a=p.parse_args();repo=a.repo.resolve();folder=ROOT/"results/angular_v9_balanced"
    result=json.loads((folder/"summary.json").read_text());cfg=result["config"]
    assert cfg==json.loads((folder/"config.json").read_text())
    assert follow.primary.sha(ROOT/"docs/PROTOCOL_V9_KERNEL_FOLLOWUP.md")==cfg["protocol_sha256"]
    for name,digest in cfg["source_sha256"].items():
        p=ROOT/("analysis" if name.startswith("validate_") else "packaging")/name
        assert follow.primary.sha(p)==digest
    prior=set()
    for base,dirname,key in [(repo,"validation_v7","data_sha256"),(repo,"angular_v8","source_data_sha256"),(repo,"angular_v8_fourth","source_data_sha256"),(ROOT,"angular_v9","source_data_sha256")]:
        for p in (base/"results"/dirname).glob("*/*.json"):
            r=json.loads(p.read_text())
            if key in r:prior.add(r[key])
    assert len(prior)==6400,len(prior)
    unique={};records={};tasks=[];total=0
    for name,summary in result["cells"].items():
        family,n,q=summary["family"],summary["n"],summary["q"];rows=[]
        for i in range(cfg["datasets"]):
            r=json.loads((folder/name/f"{i:05}.json").read_text());x,base=follow.prepare(family,n,q,i)
            for k,v in base.items():assert r[k]==v
            assert hashlib.sha256(follow.nv.normalize(x).astype("<f8").tobytes()).hexdigest()==r["data_sha256"]
            assert r["source_data_sha256"] not in prior
            key=(family,n,i)
            if key in unique:assert unique[key]==r["source_data_sha256"]
            unique[key]=r["source_data_sha256"];total+=1
            assert r["p"]["KB"]==min(1.,2*min(r["p"]["KE"],r["p"]["KO"]))
            assert all(0<v<=1 and abs(v*200-round(v*200))<1e-10 for v in r["p"].values())
            rows.append(r)
        for k,v in follow.summarize(rows).items():assert summary[k]==v,(name,k)
        assert summary==json.loads((folder/name/"summary.json").read_text())
        records[name]=rows;tasks.append((family,n,q,0,cfg))
    assert len(set(unique.values()))==len(unique)
    combined={}
    for family in cfg["families"]:
        for n in cfg["n"]:
            byq=[records[f"{family}_n{n}_q{q:02}"] for q in cfg["q"]];arms={}
            for arm in cfg["arms"]:
                values=[min(1.,3*min(rows[i]["p"][arm] for rows in byq)) for i in range(cfg["datasets"])]
                count=sum(v<=.05 for v in values);arms[arm]=dict(count=count,rate=count/len(values),ci=follow.primary.interval(count,len(values)))
            combined[f"{family}_n{n}"]=dict(family=family,n=n,arms=arms)
    replayed=[]
    if a.replay:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs={pool.submit(follow.one,t):f"{t[0]}_n{t[1]}_q{t[2]:02}" for t in tasks}
            for job in as_completed(jobs):
                name=jobs[job];assert job.result()==records[name][0],name;replayed.append(name)
    audit=dict(projected_analyses=total,unique_source_datasets=len(unique),no_overlap_with_prior=True,
        total_unique_datasets=len(prior|set(unique.values())),source_and_data_hashes_verified=True,
        summaries_recomputed=True,combined_three_projection=combined,replayed_cells=sorted(replayed))
    follow.primary.dump(folder/"audit.json",audit)
    print(json.dumps({k:v for k,v in audit.items() if k!="combined_three_projection"},indent=2))
if __name__=="__main__":main()
