#check S16 provenance, pairing, projection diagnostics and rates
from pathlib import Path
import hashlib,json,math
from scipy.stats import binomtest
BASE=Path(__file__).resolve().parent;OUT=BASE/"results"
cfg=json.loads((OUT/"config.json").read_text());summary=json.loads((OUT/"summary.json").read_text())
rows=[json.loads(p.read_text()) for p in (OUT/"records").glob("*.json")]
assert len(rows)==1800 and not summary["errors"]
assert len({r["source_data_sha256"] for r in rows})==600
cal={r["source_seed"] for r in rows if r["group"]=="calibration_null"}
evals={r["source_seed"] for r in rows if r["group"]!="calibration_null"}
assert len(cal)==200 and len(evals)==200 and not cal.intersection(evals)
for name,expected in cfg["source_sha256"].items():
    path=BASE/name if name=="run_validation.py" else BASE/"modules"/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
lookup={(r["group"],r["projection"],r["index"]):r for r in rows}
for g in ("calibration_null","evaluation_null","leading_mixture"):
    for i in range(200):
        triplet=[lookup[g,k,i] for k in ("nested5","nested10","oracle5")]
        assert len({r["source_data_sha256"] for r in triplet})==1
        assert triplet[0]["retained_axis_length_squared"]<=triplet[1]["retained_axis_length_squared"]+1e-12
        assert triplet[0]["projected_squared_separation"]<=triplet[1]["projected_squared_separation"]+1e-10
        if g=="leading_mixture":assert abs(triplet[2]["projected_squared_separation"]-16)<1e-9
for r in rows:
    assert all(abs(v*200-round(v*200))<1e-8 for v in r["p"].values())
    assert r["shape_fit"]["whitened_moment_residual"]<1e-7
    assert r["max_reference_residual"]<1e-7
comparisons=[]
for mode in ("nominal","calibrated"):
    for arm in ("W","D4","K","E"):
        for a,b in [("nested5","nested10"),("nested5","oracle5"),("nested10","oracle5")]:
            cut=lambda kind:.05 if mode=="nominal" else summary["cells"]["leading_mixture_"+kind]["arms"][arm]["independently_calibrated_p_cutoff"]
            aa=[lookup["leading_mixture",a,i]["p"][arm]<=cut(a) for i in range(200)]
            bb=[lookup["leading_mixture",b,i]["p"][arm]<=cut(b) for i in range(200)]
            left=sum(x and not y for x,y in zip(aa,bb));right=sum(y and not x for x,y in zip(aa,bb));discordant=left+right
            diff=(right-left)/200;se=math.sqrt(max(0,(discordant/200-diff*diff)/199))
            comparisons.append(dict(mode=mode,arm=arm,a=a,b=b,a_only=left,b_only=right,
                difference_b_minus_a=diff,paired_normal95=[diff-1.959964*se,diff+1.959964*se],
                exact_p=binomtest(left,discordant,.5).pvalue if discordant else 1.))
running=0
for rank,r in enumerate(sorted(comparisons,key=lambda x:x["exact_p"])):
    running=max(running,min(1,(len(comparisons)-rank)*r["exact_p"]));r["Holm_p_over_24_exploratory_projection_comparisons"]=running
result=dict(passed=True,analyses=len(rows),distinct_source_arrays=600,distinct_evaluation_base_noise_seeds=200,
    independent_calibration_noise_seeds=200,checks=["source checksums","calibration/evaluation independence","paired projections","nested signal metrics","oracle separation","Monte Carlo p grid","Tyler residuals"],
    max_fitting_residual=max(max(r["shape_fit"]["whitened_moment_residual"],r["max_reference_residual"]) for r in rows),
    paired_projection_comparisons=comparisons)
(OUT/"verification.json").write_text(json.dumps(result,indent=2))
print("PASS: all 1,800 records and paired-design checks; maximum residual",result["max_fitting_residual"])
for name,c in summary["cells"].items():
    if name.startswith("calibration"):continue
    print(name,{a:(v["nominal_rate"],v["calibrated_rate"]) for a,v in c["arms"].items()})
