import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
import recorded_controls as rc

def test_stored_spectra_reach_their_targets():
    sp=rc.spectra(20260926)
    for cell,(lam,c) in ((k,v) for k,v in sp.items() if k!="reversal"):
        assert lam.shape==(119,) and abs(c.sum()-1)<1e-10 and np.all(c>0)
    pr=lambda l:l.sum()**2/(l**2).sum()
    assert pr(sp["reversal"])==pr(sp["cell3"][0])

def test_arms_share_data_and_seeds():
    rc.init(rc.spectra(98765))
    r=rc.one_a((98765,"cell1",0,9))
    assert "error" not in r and set(r["p"])==set(rc.ARMS)
    assert r["data_seed"]!=r["reference_seed"]

def test_reversal_record_retests_only_discordant():
    rc.init(rc.spectra(98765))
    r=rc.one_b((98765,0,0,9,19))
    assert "error" not in r
    disc=(r["p"]["M"]<.05)!=(r["p"]["C"]<.05)
    assert (r["retest"] is not None)==disc
