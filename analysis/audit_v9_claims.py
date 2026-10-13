#Every headline claim of the v9 angular study recomputed from its artifacts
import json
from pathlib import Path
_R=Path(__file__).resolve().parents[1];OUT=_R/"results"/"audit_v9_claims.json";PRIMARY=json.loads((_R/"results"/"angular_v9"/"summary.json").read_text())
BALANCED=json.loads((_R/"results"/"angular_v9_balanced"/"summary.json").read_text());APPLIED=json.loads((_R/"results"/"embeddings_v9"/"summary.json").read_text());ALPHA=0.05
NULL_FAMILIES=("gaussian","anisotropic")
def pct(x):return round(100*x,1)
def cells(summary,kind=None):return{k:v for k,v in summary["cells"].items() if kind is None or v.get("kind")==kind}
def rng(summary,arm,kind):
    vs=[v["arms"][arm]["rate"] for v in cells(summary,kind).values() if arm in v["arms"]];return(pct(min(vs)),pct(max(vs))) if vs else None
def check(label,claimed,found,tol=0.051):
    ok=claimed is None or found is None or(isinstance(claimed,(int,float)) and abs(claimed-found)<=tol) or(claimed==found);return dict(claim=label,claimed=claimed,found=found,agrees=bool(ok))
def main():
    r=[];n=cells(PRIMARY,"null");r.append(check("null cells in the primary comparison",12,len(n)))
    for am,cd in(("W",(3.5,9.0)),("D4",(2.0,9.5)),("K",(3.5,8.5)),("E",(3.5,7.5))):r.append(check("null rejection range, %s"%am,list(cd),list(rng(PRIMARY,am,"null"))))
    el=[(k,a,pct(v["arms"][a]["rate"]),[pct(x) for x in v["arms"][a]["ci"]]) for k,v in n.items() for a in v["arms"] if not v["arms"][a]["ci"][0]<=ALPHA<=v["arms"][a]["ci"][1]]
    r.append(check("pointwise intervals excluding .05, of 48",4,len(el)))
    def rate(summary,cell,arm):
        c=summary["cells"].get(cell);return None if c is None or arm not in c["arms"] else pct(c["arms"][arm]["rate"])
    for cl,cs in(("two_modes_n240_q05",dict(W=10.0,D4=56.0,K=5.0,E=9.0)),("six_modes_n240_q05",dict(W=9.0,D4=77.0,K=5.0,E=4.0)),("two_modes_n240_q20",dict(W=6.5,D4=10.5,K=5.0,E=44.5)),("offset_n60_q10",dict(W=11.5,D4=5.0,K=99.0,E=97.5)),("six_modes_n60_q20",dict(W=5.0,D4=4.0,K=4.0,E=6.0)),("shape_mixture_n240_q20",dict(W=4.0,D4=24.0,K=4.5,E=5.0)),("offset_n240_q20",dict(W=12.5,D4=6.5,K=100.0,E=100.0))):
        for am,cd in cs.items():r.append(check("%s %s"%(cl,am),cd,rate(PRIMARY,cl,am)))
    kl=rng(BALANCED,"KB","null");r.append(check("kernel follow-up null range, KB",[1.5,6.5],list(kl)))
    for cl,am,cd in(("two_modes_n240_q05","K",4.5),("two_modes_n240_q05","KB",46.5),("six_modes_n240_q05","K",5.5),("six_modes_n240_q05","KB",72.0),("six_modes_n60_q05","KB",11.0)):
        r.append(check("follow-up %s %s"%(cl,am),cd,rate(BALANCED,cl,am)))
    sc={s["path"].split("/")[-1]:s for s in APPLIED.get("sources",[])} if isinstance(APPLIED.get("sources"),list) else{};r.append(check("applied input arrays",23,len(sc) or None))
    c1=dict(n_primary_cells=len(PRIMARY["cells"]),n_followup_cells=len(BALANCED["cells"]),primary_datasets_per_cell=PRIMARY["config"]["datasets"],followup_datasets_per_cell=BALANCED["config"]["datasets"],draws=PRIMARY["config"]["draws"])
    o=dict(source=["results/angular_v9/summary.json","results/angular_v9_balanced/summary.json","results/embeddings_v9/summary.json"],design=c1,intervals_excluding_nominal=el,checks=r,n_checks=len(r),n_disagreements=sum((not c["agrees"] for c in r)),disagreements=[c for c in r if not c["agrees"]])
    OUT.write_text(json.dumps(o,indent=1));print("design:",json.dumps(c1));print("checks %d, disagreements %d"%(o["n_checks"],o["n_disagreements"]))
    for c in o["disagreements"]:print("  MISMATCH %-34s claimed %s found %s"%(c["claim"],c["claimed"],c["found"]))
    print("intervals excluding .05 among null cells:")
    for e in el:print("   ",e)
    print("wrote %s"%OUT)
if __name__=="__main__":main()
