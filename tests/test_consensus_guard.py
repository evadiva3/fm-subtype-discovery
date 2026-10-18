import importlib.util,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def mod():
 p=R/"experiments/ds002748_mdd/propagated_null";o=sys.path[:];s={k:sys.modules.get(k) for k in ("ov_common","nullcal")}
 try:
  sys.path.insert(0,str(p))
  for k in s:sys.modules.pop(k,None)
  a=importlib.util.spec_from_file_location("cons_guard",p/"e11_consensus.py");m=importlib.util.module_from_spec(a);a.loader.exec_module(m);return m
 finally:
  sys.path[:]=o
  for k,v in s.items():
   if v is None:sys.modules.pop(k,None)
   else:sys.modules[k]=v
def test_impossible_guard_is_not_reported_as_satisfied():
 x=np.array([0,.01,.4,.45,1,1.1]);c=np.exp(-abs(x[:,None]-x));l,k,s,g=mod().consensus_partition(c,4,range(2,5));assert l is not None;assert not g;assert min(np.unique(l,return_counts=True)[1])<4
def test_feasible_guard_is_satisfied():
 x=np.array([0,.01,.02,.03,2,2.01,2.02,2.03]);c=np.exp(-abs(x[:,None]-x));l,k,s,g=mod().consensus_partition(c,3,range(2,5));assert g;assert min(np.unique(l,return_counts=True)[1])>=3
