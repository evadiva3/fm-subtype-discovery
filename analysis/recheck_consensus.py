from pathlib import Path
import os,sys,json,hashlib,argparse
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
R=Path(__file__).resolve().parents[1];P=R/"experiments/ds002748_mdd/propagated_null";sys.path.insert(0,str(P))
import e11_consensus as e
a=argparse.ArgumentParser();a.add_argument("--output",required=True);p=Path(a.parse_args().output);assert not p.exists(),p
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={"purpose":"Corrected consensus guard on the saved depression label vectors; no encoder retraining","seed":e.SEED,"bootstrap_n":e.N_BOOT,"source_sha256":h(P/"e11_consensus.py"),"cells":{},"inputs":[]}
for n,f,pat in [("fixed_arch",e.load_fixed_arch,"results/mdd/fixed_arch_control/runs/seed*.json"),("search_vary",e.load_search_vary,"results/mdd/multirun_v2/by_run/run*_labels.csv")]:
 l,ids,nf=f();assert l is not None and l.shape==(20,51),(n,None if l is None else l.shape)
 if n=="fixed_arch":
  js=[json.loads(q.read_text()) for q in sorted(R.glob(pat)) if "labels"in json.loads(q.read_text())]
  assert all(q.get("subject_ids")==ids for q in js)
 d["cells"][n]=e.analyse(n,l,ids,nf)
 assert d["cells"][n]["guard_satisfied"] and min(d["cells"][n]["consensus_sizes"])>=d["cells"][n]["guard"]
 d["inputs"].extend({"path":q.relative_to(R).as_posix(),"sha256":h(q)} for q in sorted(R.glob(pat)))
d["limitations"]=["No fibromyalgia label vectors available for this analysis","Bootstrap resamples training runs and does not calibrate a population-subtype test","Historical outputs are preserved; this is a corrected analysis of the same saved labels"]
p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({n:{k:c[k] for k in ("n_subjects","n_runs","consensus_k","consensus_sizes","guard","guard_satisfied","ari_bootstrap_consensus_mean","ari_pairwise_runs_mean","bootstrap_guard_failures")} for n,c in d["cells"].items()},indent=2))
