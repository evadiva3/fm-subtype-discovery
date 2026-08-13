import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

_R=Path(__file__).resolve().parents[1]
if str(_R) not in sys.path:
    sys.path.insert(0,str(_R))

from config import config

RES=Path(config.resultsRoot)

KNOWN={
 "sensitivity_percentile_sweep.csv":("other-checkpoint",
  "four retrains at edge percentiles 75/80/85/90; each row is its own model"),
 "baseline_comparison.csv":("derived",
  "canonical copy with the group-ICA row recomputed locally"),
 "null_corrected/phase5_2_fm20run_corrected.csv":("other-checkpoint",
  "20-run protocol, prior search space, 1000 draws"),
 "null_corrected/phase5_2_fm20run_corrected_20000.csv":("other-checkpoint",
  "same 20 runs at 20000 draws, prior search space"),
 "null_corrected/phase5_3_summary_20000.json":("other-checkpoint",
  "summary of the above; source of the PROV 20-run figures"),
 "null_corrected/phase4_canonical_k4.json":("canonical",
  "canonical checkpoint, misspecified vs corrected at 20000 draws"),
 "null_corrected/null_progression_1000.json":("canonical",
  "four null constructions on the canonical checkpoint"),
 "null_corrected/null_progression_1000.csv":("canonical",
  "four null constructions on the canonical checkpoint"),
 "null_corrected/null_selected_k_20000.json":("canonical",
  "null selected-k distribution, canonical, 20000 draws"),
 "null_corrected/_table4_20000_incremental.csv":("other-checkpoint",
  "scratch from the 20-run recompute; superseded, do not cite"),
 "null_corrected/phase5_2_fm20run_20000_PARTIAL_runs1-8.csv":("other-checkpoint",
  "PARTIAL runs 1-8 from an interrupted recompute; do not cite"),
 "separation_ratio_conversion.csv":("canonical",
  "delta to separation ratio on the canonical geometry"),
 "separation_ratio_conversion.json":("canonical",
  "delta to separation ratio on the canonical geometry"),
 "end_to_end_positive_control.json":("canonical",
  "condition-graph clustering with the canonical encoder"),
 "type1_surface.json":("canonical",
  "type I error by geometry; anchor cell uses the canonical covariance"),
 "stability_summary.csv":("other-checkpoint",
  "14 independently searched architectures; none is the canonical checkpoint"),
}

META={"PROVENANCE.json","PROVENANCE.md","CANONICAL_MANIFEST.json"}

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):
            h.update(c)
    return h.hexdigest()

def fingerprint():
    ps={"checkpoint":Path(config.jointCheckpointPath),
        "bestParams":Path(config.raySavePath),
        "embeddings":Path(config.clusterOutput)/"trained_fm_embeddings.npy",
        "labels":Path(config.kLabelPath)}
    f={}
    for n,p in ps.items():
        try:
            rel=str(p.relative_to(_R))
        except ValueError:
            rel=str(p)          
        f[n]={"path":rel,"sha256":sha(p) if p.exists() else None,
              "exists":p.exists()}
    bp=ps["bestParams"]
    if bp.exists():
        d=json.loads(bp.read_text())
        f["corrected_search"]="batchSize" not in d
        f["arch"]={k:list(v.values())[0] if isinstance(v,dict) else v for k,v in d.items()}
    try:
        import torch
        ck=torch.load(ps["checkpoint"],map_location="cpu",weights_only=False)
        f["norm_stats"]=bool(isinstance(ck,dict) and "nodeMean" in ck and "nodeStd" in ck)
    except Exception as e:
        f["norm_stats"]=f"unchecked: {type(e).__name__}"
    return f

def manifest():
    p=RES/"CANONICAL_MANIFEST.json"
    return json.loads(p.read_text()).get("files",{}) if p.exists() else {}

def audit():
    f=fingerprint()
    mn=manifest()
    rows=[]
    for p in sorted(RES.rglob("*")):
        if not p.is_file() or p.name.startswith(".") or p.suffix in (".log",".out",".png",".sh",".svg"):
            continue
        rl=str(p.relative_to(RES))
        if rl in META:
            continue
        dg=sha(p)
        t,nt=KNOWN.get(rl,(None,None))
        if t is None and mn:
            a=mn.get(f"results/{rl}")
            if a==dg:
                t,nt="canonical-run","byte-identical to the canonical archive"
            elif a is not None:
                t,nt="derived","in the archive but modified since"
        if t is None:
            if rl.startswith("handoff_20260802/propagated_null"):
                t,nt="canonical","propagated null against the canonical checkpoint"
            elif rl.startswith("handoff_20260802/calibration_paired"):
                t,nt="canonical","paired type I calibration on the canonical geometry"
            elif rl.startswith("handoff_20260802/feature_validity"):
                t,nt="derived","computed from connectivity, upstream of any checkpoint"
            elif rl.startswith("handoff_20260802/crossrun_ari"):
                t,nt="other-checkpoint","agreement across the 14 stability runs"
            elif rl.startswith("handoff_20260802/audit_gap_fills"):
                t,nt="canonical","gap fills on the canonical checkpoint"
            elif rl.startswith("handoff_20260802/figures"):
                t,nt="derived","figures from the canonical checkpoint"
            elif rl.startswith("mdd/"):
                t,nt="other-checkpoint","depression cohort, own canonical checkpoint"
            elif rl.startswith("calibration/"):
                t,nt="canonical","calibration on the canonical geometry"
            elif rl.startswith("figures/"):
                t,nt="derived","figure cache"
            else:
                t,nt="UNVERIFIED","no rule; verify before citing"
        rows.append({"path":rl,"sha256":dg[:16],
                     "mtime":datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                     "provenance":t,"note":nt})
    d={"generated":datetime.now(timezone.utc).isoformat(timespec="seconds"),
       "fingerprint":f,"artifacts":rows}
    (RES/"PROVENANCE.json").write_text(json.dumps(d,indent=2))
    ar=f.get("arch",{})
    ln=["# Artifact provenance","",f"Generated {d['generated']}.","",
        "## Canonical checkpoint","",
        f"- checkpoint `{(f['checkpoint']['sha256'] or '')[:16]}`",
        f"- bestParams `{(f['bestParams']['sha256'] or '')[:16]}`",
        f"- embeddings `{(f['embeddings']['sha256'] or '')[:16]}`",
        f"- corrected search space: **{f.get('corrected_search')}**",
        f"- train-split norm stats: **{f.get('norm_stats')}**",
        f"- arch: dModel {ar.get('dModel')}, heads {ar.get('heads')}, "
        f"layers {ar.get('layers')}, temp {ar.get('ntXentTemp')}",""]
    ct={}
    for r in rows:
        ct[r["provenance"]]=ct.get(r["provenance"],0)+1
    ln+=["## Summary",""]+[f"- {k}: {v}" for k,v in sorted(ct.items())]+[""]
    ln+=["## Artifacts","","| file | sha256 | modified | provenance | note |","|---|---|---|---|---|"]
    for r in rows:
        ln.append(f"| `{r['path']}` | `{r['sha256']}` | {r['mtime']} | "
                  f"**{r['provenance']}** | {r['note']} |")
    (RES/"PROVENANCE.md").write_text("\n".join(ln)+"\n")
    return d

def stamp(out,ins,drv,extra=None):
    out=Path(out)
    r={"output":str(out),"written":datetime.now(timezone.utc).isoformat(timespec="seconds"),
       "driver":drv,"inputs":[{"path":str(Path(i)),"sha256":sha(Path(i))}
                              for i in ins if Path(i).exists()],
       "fingerprint":fingerprint()}
    if extra:
        r["extra"]=extra
    s=out.with_suffix(out.suffix+".prov.json")
    s.write_text(json.dumps(r,indent=2))
    return s

if __name__=="__main__":
    d=audit()
    f=d["fingerprint"]
    for n in ("checkpoint","bestParams","embeddings","labels"):
        e=f.get(n,{})
        print(f"{n} {(e.get('sha256') or 'MISSING')[:16]}")
    print(f"corrected search {f.get('corrected_search')}")
    print(f"norm stats {f.get('norm_stats')}")
    c={}
    for r in d["artifacts"]:
        c[r["provenance"]]=c.get(r["provenance"],0)+1
    for k,v in sorted(c.items()):
        print(f"  {k}: {v}")
    print(f"wrote {RES}/PROVENANCE.md")
