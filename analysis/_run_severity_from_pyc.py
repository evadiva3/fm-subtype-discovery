# just a run file
import sys, warnings, marshal
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT/"src", ROOT/"models", ROOT/"analysis", ROOT/"preprocessing"):
    sys.path.insert(0, str(p))
import subject_filter
_orig=subject_filter.get_included_subjects
_patched=lambda *a, **k: [s for s in _orig(*a, **k) if s != "sub-008"]
subject_filter.get_included_subjects=_patched
import dataset as dataset_mod
dataset_mod.get_included_subjects=_patched
import models.dataset as _md
_md.get_included_subjects=_patched
PYC=ROOT/"analysis"/"__pycache__"/"severity_gradient_regression.cpython-314.pyc"
with open(PYC, "rb") as fh:
    fh.read(16)
    code=marshal.load(fh)

g={"__name__": "__main__", "__file__": str(ROOT/"analysis"/"severity_gradient_regression.py")}
exec(code, g)
