from pathlib import Path
import json,subprocess,sys,tempfile
from moves import here,same
R=Path(__file__).resolve().parents[1];P=R/"results/recorded/s6_s9";a=json.loads((P/"consensus_guard_corrected.json").read_text())
assert same("experiments/ds002748_mdd/propagated_null/e11_consensus.py",a["source_sha256"])

def norm(x):
    #the saved record has the old folder names and the hash of the original e11_consensus.py
    x={k:v for k,v in x.items() if k!="source_sha256"}
    x["inputs"]=[dict(i,path=here(i["path"]).relative_to(R).as_posix()) for i in x["inputs"]]
    return x

with tempfile.TemporaryDirectory(prefix="consensus_check_",dir=P) as t:
 p=Path(t)/"result.json";subprocess.run([sys.executable,str(R/"analysis/recheck_consensus.py"),"--output",str(p)],cwd=R,check=True,stdout=subprocess.DEVNULL);b=json.loads(p.read_text());assert norm(a)==norm(b),"Corrected consensus replay differs from the saved record"
d={"passed":True,"protocols":2,"subjects_per_protocol":51,"runs_per_protocol":20,"bootstrap_samples_replayed":2000,"bootstrap_guard_failures":sum(c["bootstrap_guard_failures"] for c in b["cells"].values()),"exact_replay":True}
(R/"reproducibility/audits/consensus_replay.json").write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
