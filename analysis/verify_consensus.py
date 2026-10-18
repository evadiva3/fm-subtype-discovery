from pathlib import Path
import json,subprocess,sys,tempfile
R=Path(__file__).resolve().parents[1];P=R/"results/revision_20260923";a=json.loads((P/"consensus_guard_corrected.json").read_text())
with tempfile.TemporaryDirectory(prefix="consensus_check_",dir=P) as t:
 p=Path(t)/"result.json";subprocess.run([sys.executable,str(R/"analysis/recheck_consensus.py"),"--output",str(p)],cwd=R,check=True,stdout=subprocess.DEVNULL);b=json.loads(p.read_text());assert a==b,"Corrected consensus replay differs from the saved record"
d={"passed":True,"protocols":2,"subjects_per_protocol":51,"runs_per_protocol":20,"bootstrap_samples_replayed":2000,"bootstrap_guard_failures":sum(c["bootstrap_guard_failures"] for c in b["cells"].values()),"exact_replay":True}
(R/"reproducibility/audits/consensus_replay.json").write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
