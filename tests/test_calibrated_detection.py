import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
from summarize_detection import cutoff,reject
from calibrated_detection import one,sd
from summarize_detection import summarize
import json

def test_tied_cutoff_has_exact_empirical_expected_size():
    p=np.r_[np.full(10,.005),np.full(80,.01),np.full(910,.5)]
    t,r=cutoff(p)
    assert t==.01 and r==.5
    assert np.mean(p<t)+r*np.mean(p==t)==.05
    assert reject(np.array([.005,.01,.01,.5]),np.array([.9,.49,.5,0.]),t,r).tolist()==[True,True,False,False]

def test_all_ties_still_allow_five_percent():
    t,r=cutoff(np.ones(1000))
    assert t==1. and r==.05

def test_nested_sources_and_independent_calibration():
    a=one((98765,0,5,1,0,2))
    b=one((98765,0,10,1,0,2))
    c=one((98765,0,5,0,0,2))
    assert not any("error" in r for r in [a,b,c])
    assert a["source_sha256"]==b["source_sha256"]
    assert a["source_sha256"]!=c["source_sha256"]
    assert a["projected_sha256"]!=b["projected_sha256"]
    assert len({sd(98765,k,0,5,1,0) for k in [2,3,4]})==3

def test_bootstrap_preserves_identical_method_pairing(tmp_path):
    cfg=dict(qs=[5],groups={"calibration":100,"null":100,"leading2":100},bootstrap_resamples=100)
    (tmp_path/"config.json").write_text(json.dumps(cfg),encoding="utf-8")
    rows=[]
    for s in range(2):
        for g in cfg["groups"]:
            for i in range(100):
                p=.005 if g!="calibration" or i<10 else .5
                rows.append(dict(spectrum=s,q=5,group=g,index=i,p={a:p for a in ("W","D4","K","E")},boundary_uniform=(i+.5)/100,guard_failed=False,reference_guard_failures=0,projected_squared_half_separation=0.,retained_axis_squared=1.))
    (tmp_path/"records.jsonl").write_text('\n'.join(map(json.dumps,rows))+'\n',encoding="utf-8")
    z=summarize(tmp_path)
    assert json.loads(json.dumps(z))==z
    assert isinstance(z["cells"]["0_q5_null"]["arms"]["W"]["heldout_tolerance_03_07_supported"],bool)
    for c in z["comparisons"].values():
        for v in c.values():
            assert v["rate_difference"]==0.
            assert v["two_stage_bootstrap95"]==[0.,0.]
    v=z["cells"]["0_q5_null"]["arms"]["W"]
    assert v["two_stage_bootstrap95"][1]-v["two_stage_bootstrap95"][0]>.2
