#How often the sign of the re-projection shift predicts the worse construction.
import json
from pathlib import Path
_R=Path(__file__).resolve().parents[1];SRC=_R/"results"/"overnight"/"e18_boundary_surface.json";OUT=_R/"results"/"sign_rule_audit.json";BAND=0.03;GAPS=[0.0,0.02,0.05,0.1]
def agrees(v):return(v["shift"]<0)==(v["rej_C"]>v["rej_M"])
def main():
    c=json.loads(SRC.read_text())["cells"];r=[v for v in c.values() if not v.get("failed") and v.get("shift") is not None];d=[v for v in r if v["rej_C"]!=v["rej_M"]]
    o=dict(source="results/overnight/e18_boundary_surface.json",rule="a negative shift predicts that C rejects more often than M, a positive shift the reverse",indeterminate_band_in_sd=BAND,n_cells=len(r),datasets_per_cell=r[0]["n_datasets"],n_cells_with_equal_rates=len(r)-len(d),n_cells_compared=len(d),n_agree=sum((agrees(v) for v in d)),n_disagree=sum((not agrees(v) for v in d)))
    o["exceptions_inside_band"]=sum((1 for v in d if not agrees(v) and abs(v["shift_in_sd"])<BAND))
    o["by_rate_gap"]=[dict(min_gap=g,n=sum((1 for v in d if abs(v["rej_C"]-v["rej_M"])>=g)),agree=sum((1 for v in d if abs(v["rej_C"]-v["rej_M"])>=g and agrees(v)))) for g in GAPS]
    oe=[v for v in d if abs(v["shift_in_sd"])>=BAND];o["outside_band"]=dict(n=len(oe),agree=sum((agrees(v) for v in oe)))
    o["exceptions"]=sorted((dict(pr_target=v["pr_target"],n=v["n"],d=v["d"],spike=v["spike_S"],shift_in_sd=v["shift_in_sd"],rej_M=v["rej_M"],rej_C=v["rej_C"],rate_gap=abs(v["rej_C"]-v["rej_M"]),inside_indeterminate_band=abs(v["shift_in_sd"])<BAND) for v in d if not agrees(v)),key=lambda r:abs(r["shift_in_sd"]))
    b=[v for v in d if abs(v["rej_C"]-v["rej_M"])>=0.05]
    o["largest_gap_summary"]=dict(n=len(b),agree=sum((agrees(v) for v in b)),exceptions_all_inside_band=all((abs(v["shift_in_sd"])<BAND for v in b if not agrees(v))))
    OUT.write_text(json.dumps(o,indent=1));print(json.dumps({k:v for k,v in o.items() if k!="exceptions"},indent=1));print("wrote %s"%OUT)
if __name__=="__main__":main()
