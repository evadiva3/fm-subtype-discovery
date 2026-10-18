import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
import mc_calibrated_sensitivity as m
import sample_size_detection as s
import grid_recorded as g

def test_s24_alternatives_are_centered_and_sized():
    perm=np.random.default_rng(0).permutation(m.N)
    for name in m.G[2:]:
        M,lab=m.means(name,perm)
        w=np.bincount(lab)/m.N
        assert np.allclose(w@M,0)
        assert np.bincount(lab).tolist()==([14,14] if name.startswith("two") else [10,9,9])
    assert m.means("null",perm)==(None,None)

def test_s24_record_runs_with_both_arms():
    r=m.one((98765,m.G.index("two1.0"),0,2))
    assert "error" not in r and set(r["p"])=={"M","C"} and r["standardized_between"]>0

def test_s25_four_group_sizes_follow_protocol():
    assert s.sizes("tetra",50)==[14,12,12,12] and s.sizes("tetra",200)==[50]*4
    assert s.sizes("tetra",28)==[7]*4 and s.sizes("tri",50)==[18,16,16]

def test_s26_uses_saved_tuned_cells():
    assert len(g.CELLS)==30 and {c["n"] for c in g.CELLS}=={28,60,119,180,240}
    r=g.one((98765,0,0,2))
    assert "error" not in r and r["n"]==g.CELLS[0]["n"]
