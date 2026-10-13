# every summary, verify hashes, and replay selected complete v9 analyses.
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[name]="1"
import argparse,hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import validate_angular_gof_v9 as study
import analyze_saved_gof_v9 as applied
ROOT=Path(__file__).resolve().parents[1]
def verify_sources(cfg,protocol):
    assert study.sha(ROOT/"docs"/protocol)==cfg["protocol_sha256"]
    for name,digest in cfg["source_sha256"].items():
        p=ROOT/("analysis" if name.startswith(("validate_","analyze_")) else "packaging")/name
        assert study.sha(p)==digest,(name,"source changed")
def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,default=ROOT);p.add_argument("--workers",type=int,default=8)
    p.add_argument("--replay",action="store_true");a=p.parse_args();repo=a.repo.resolve()
    folder=ROOT/"results/angular_v9";result=json.loads((folder/"summary.json").read_text());cfg=result["config"]
    assert cfg==json.loads((folder/"config.json").read_text());verify_sources(cfg,"PROTOCOL_V9.md")
    prior=set()
    for dirname,key in [("validation_v7","data_sha256"),("angular_v8","source_data_sha256"),("angular_v8_fourth","source_data_sha256")]:
        for file in (repo/"results"/dirname).glob("*/*.json"):
            r=json.loads(file.read_text())
            if key in r:prior.add(r[key])
    assert len(prior)==4000,len(prior)
    unique={};records={};tasks=[];total=0
    for name,summary in result["cells"].items():
        family,n,q=summary["family"],summary["n"],summary["q"];rows=[]
        for i in range(cfg["datasets"]):
            r=json.loads((folder/name/f"{i:05}.json").read_text());x,base=study.prepare(family,n,q,i)
            for k,v in base.items():assert r[k]==v,(name,i,k)
            assert hashlib.sha256(study.nv.normalize(x).astype("<f8").tobytes()).hexdigest()==r["data_sha256"]
            assert r["source_data_sha256"] not in prior
            key=(family,n,i)
            if key in unique:assert unique[key]==r["source_data_sha256"]
            unique[key]=r["source_data_sha256"]
            assert all(0<p<=1 and abs(p*(cfg["draws"]+1)-round(p*(cfg["draws"]+1)))<1e-10 for p in r["p"].values())
            rows.append(r);total+=1
        for k,v in study.summarize(rows).items():assert summary[k]==v,(name,k)
        assert summary==json.loads((folder/name/"summary.json").read_text())
        records[name]=rows;tasks.append((family,n,q,0,cfg))
    assert len(set(unique.values()))==len(unique)
    combined={}
    for family in cfg["families"]:
        for n in cfg["n"]:
            byq=[records[f"{family}_n{n}_q{q:02}"] for q in cfg["q"]];arms={}
            for arm in cfg["arms"]:
                values=[min(1.,len(byq)*min(rows[i]["p"][arm] for rows in byq)) for i in range(cfg["datasets"])]
                count=sum(v<=.05 for v in values)
                arms[arm]=dict(count=count,rate=count/len(values),ci=study.interval(count,len(values)))
            combined[f"{family}_n{n}"]=dict(family=family,n=n,arms=arms)
    empirical=json.loads((ROOT/"results/embeddings_v9/summary.json").read_text());ecfg=empirical["config"]
    verify_sources(ecfg,"PROTOCOL_V9_APPLIED.md");applied_tasks=[]
    for s in empirical["sources"]:
        assert applied.sha(repo/s["path"])==s["sha256"]
        for row in s["q_results"]:
            assert row==json.loads((ROOT/"results/embeddings_v9"/f"source{s["id"]:02}_q{row["q"]:02}.json").read_text())
            raw=np.load(repo/s["path"],allow_pickle=False)
            proj=study.an.fixed_projection(raw.shape[1],row["q"],int(row["projection_seed"]))
            x=study.nv.normalize(study.nv.normalize(raw@proj))
            assert hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()==row["data_sha256"]
            if s["id"]<3:applied_tasks.append((dict(id=s["id"],path=s["path"],sha256=s["sha256"]),row["q"],ecfg,str(repo)))
        for arm in study.gof.ARMS:assert s["combined"][arm]==min(1.,2*min(r["p"][arm] for r in s["q_results"]))
    replayed=[];applied_replayed=[]
    if a.replay:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs={pool.submit(study.one,t):f"{t[0]}_n{t[1]}_q{t[2]:02}" for t in tasks}
            for job in as_completed(jobs):
                name=jobs[job];assert job.result()==records[name][0],("replay mismatch",name);replayed.append(name)
            jobs={pool.submit(applied.one,t):(t[0]["id"],t[1]) for t in applied_tasks}
            for job in as_completed(jobs):
                sid,q=jobs[job];expected=next(r for r in empirical["sources"][sid]["q_results"] if r["q"]==q)
                assert job.result()==expected,("applied replay mismatch",sid,q);applied_replayed.append([sid,q])
    out=dict(primary_records=total,primary_unique_datasets=len(unique),prior_unique_datasets=len(prior),
        total_unique_datasets=len(prior|set(unique.values())),sources_and_datasets_verified=True,
        summaries_recomputed=True,combined_three_projection=combined,replayed_cells=sorted(replayed),
        applied_sources=len(empirical["sources"]),applied_analyses=sum(len(s["q_results"]) for s in empirical["sources"]),
        applied_replayed=sorted(applied_replayed))
    study.dump(folder/"audit.json",out)
    print(json.dumps({k:v for k,v in out.items() if k!="combined_three_projection"},indent=2))
if __name__=="__main__":main()
