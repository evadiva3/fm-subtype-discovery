#Audit of dataset overlap across the four-seed verification protocol.
import json
import sys
from itertools import combinations
from pathlib import Path
_R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(_R/"experiments"/"ds002748_mdd"/"propagated_null"))
import nullcal as nc
OVN=_R/"results"/"overnight";OUT=_R/"results"/"seed_overlap_audit.json";STRIDE=1000
def load(path):return json.loads(Path(path).read_text())
def overlap(bases,nds,stride=STRIDE):
    bs=[set((b+stride*i for i in range(nds))) for b in bases];p={"%d_vs_%d"%(bases[a],bases[b]):len(bs[a]&bs[b])/nds for a,b in combinations(range(len(bases)),2)};u=set().union(*bs)
    return dict(seed_bases=list(bases),stride=stride,datasets_per_base=nds,nominal_datasets=nds*len(bases),distinct_datasets=len(u),distinct_fraction=len(u)/(nds*len(bases)),pairwise_overlap_fraction=p,disjoint_pairs=[k for k,v in p.items() if v==0])
def check_stride(records,base_of_rep):
    o={int(g["seed"])-base_of_rep[g["replication"]]-STRIDE*int(g["idx"]) for g in records};return dict(checked=len(records),offsets=sorted(o),consistent=len(o)<=1)
def rate(c,n):return dict(events=int(c),n=int(n),rate=c/n,wilson95=list(nc._wilson(int(c),int(n))))
def reps_fallback(cell,reps_used,count_keys=("count_M","count_C")):
    rs=[cell["replications"][j] for j in reps_used];n=sum((r["n_datasets"] for r in rs));o=dict(replications=list(reps_used),seed_bases=[r["base"] for r in rs],n_datasets=n)
    if all((k in rs[0] for k in count_keys)):
        o["M"]=rate(sum((r[count_keys[0]] for r in rs)),n);o["C"]=rate(sum((r[count_keys[1]] for r in rs)),n)
    o["C_only"]=sum((r["C_only"] for r in rs));o["M_only"]=sum((r["M_only"] for r in rs));rc=[g for g in cell["discordant"] if g["replication"] in reps_used];c=len(rc)==o["C_only"]
    o["discordant_records_complete"]=c
    if c and all((g.get("deep") is not None for g in rc)):
        p=sum((bool(g["deep"]["persists"]) for g in rc));o["C_only_persisting_at_2000"]=p;o["C_only_persisting_fraction"]=p/len(rc) if rc else None
    else:o["C_only_persisting_at_2000"]="not obtainable: per-dataset records incomplete"
    return o
def audit_conly(name,path,base_key):
    s=load(path);n=s["config"]["n_datasets"];o=dict(artifact=str(Path(path).relative_to(_R)),cells={})
    for k,cl in s["cells"].items():
        bs=[r["base"] for r in cl["replications"]];b={j:r["base"] for j,r in enumerate(cl["replications"])};ov=overlap(bs,n)
        c=dict(overlap=ov,stride_check=check_stride(cl["discordant"],b),pooled_as_published=dict(total_C_only=cl["total_C_only"],total_datasets=cl["total_datasets"],deep=cl.get("deep")),base0_only=reps_fallback(cl,[0]))
        if ov["pairwise_overlap_fraction"]["%d_vs_%d"%(bs[0],bs[1])]==0:c["base0_plus_base1"]=reps_fallback(cl,[0,1])
        ss=[g["seed"] for g in cl["discordant"]];c["recorded_C_only_distinct_seeds"]=dict(recorded=len(ss),distinct=len(set(ss)),records_complete=len(ss)==cl["total_C_only"]);o["cells"][k]=c
    return o
def audit_e16():
    s=load(OVN/"e16_nwb_check.json");n=s["config"]["n_datasets"];o=dict(artifact="results/overnight/e16_nwb_check.json",cells={})
    for k,cl in s["cells"].items():
        bs=[r["base"] for r in cl["replications"]];bp={j:b for j,b in enumerate(bs)}
        c=dict(overlap=overlap(bs,n),stride_check=check_stride(cl["discordant"],bp),pooled_as_published=dict(pooled_M=cl["pooled_M"],pooled_M_ci=cl["pooled_M_ci"],pooled_C=cl["pooled_C"],pooled_C_ci=cl["pooled_C_ci"],total_C_only=cl["total_C_only"],total_M_only=cl["total_M_only"],deep=cl["deep"]),base0_only=reps_fallback(cl,[0]),base0_plus_base1=reps_fallback(cl,[0,1]))
        ss=[g["seed"] for g in cl["discordant"]];d={}
        for g in cl["discordant"]:d.setdefault(g["seed"],[]).append(bool(g["deep"]["persists"]))
        c["C_only_by_distinct_dataset"]=dict(recorded=len(ss),distinct_datasets=len(d),distinct_persisting_in_every_replication=sum((all(v) for v in d.values())),distinct_persisting_in_any_replication=sum((any(v) for v in d.values())))
        o["cells"][k]=c
    return o
def audit_e9():
    s=load(OVN/"e9_calibration_seeds.json");c=s["config"];r=[s["cells"]["rep%d"%j] for j in range(c["n_replications"])]
    return dict(artifact="results/overnight/e9_calibration_seeds.json",overlap=overlap(c["seed_bases"],c["n_datasets"]),published=dict(C_rates=s["summary"]["C_rates"],C_mean=s["summary"]["C_mean"],M_mean=s["summary"]["M_mean"],M_sd=s["summary"]["M_sd"]),base0_only=dict(M=rate(r[0]["count_M"],r[0]["n_datasets"]),C=rate(r[0]["count_C"],r[0]["n_datasets"])),note="consecutive bases are 100,000 apart with 300 datasets at stride 1,000, so adjacent replications share two thirds of their datasets and only the pairs listed under disjoint_pairs are independent")
def audit_popcov(path):
    s=load(path);n=s["config"]["n_datasets"];o=dict(artifact=str(Path(path).relative_to(_R)),cells={})
    for k,c in s["cells"].items():
        v=c.get("verification") or{}
        if not v.get("triggered"):continue
        r=v["replications"];p={}
        for t in("est","known"):
            p[t]=dict(M=sum((r["rej_M_"+t]*r["n_datasets"] for r in r))/v["total_datasets"],C=sum((r["rej_C_"+t]*r["n_datasets"] for r in r))/v["total_datasets"]);p[t]["C_minus_M"]=p[t]["C"]-p[t]["M"]
        r0=r[0]
        b0={t:dict(M=rate(round(r0["rej_M_"+t]*r0["n_datasets"]),r0["n_datasets"]),C=rate(round(r0["rej_C_"+t]*r0["n_datasets"]),r0["n_datasets"]),C_minus_M=r0["rej_C_"+t]-r0["rej_M_"+t],C_only=r0["C_only_"+t],M_only=r0["M_only_"+t]) for t in("est","known")}
        o["cells"][k]=dict(overlap=overlap(v["seed_bases"],n),pooled_as_published=p,base0_only=b0,deep_as_published=v.get("deep"))
    return o
def main():
    r=dict(purpose="dataset overlap across four-seed verification replications",seed_rule="dataset i of replication j uses base_j + 1000*i + cell_offset",e1_conly_check=audit_conly("e1",OVN/"e1_conly_check.json","extra_seed_bases"),e2_conly_check=audit_conly("e2",OVN/"e2_conly_check.json","extra_bases"),e16_nwb_check=audit_e16(),e9_calibration_seeds=audit_e9(),popcov_control=audit_popcov(_R/"results"/"popcov_control.json"),popcov_control_cells567=audit_popcov(_R/"results"/"popcov_control_cells567.json"),e2_spectra_seeds="results/overnight/e2_spectra_seeds.json does not exist; the per-seed verification behind the reversal is e2_conly_check.json")
    OUT.write_text(json.dumps(r,indent=1));print("wrote %s"%OUT)
if __name__=="__main__":main()
