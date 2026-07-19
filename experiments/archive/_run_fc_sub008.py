#to resolve a problem on my computer where i coundn't find sub-008
#please ignore.


import sys, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src", ROOT / "models", ROOT / "analysis", ROOT / "preprocessing"):
    sys.path.insert(0, str(p))
from config import config
import compute_fc_matrices as cfc
SUBJ="sub-008"
subj_dir=Path(config.subjectDataFolder) / SUBJ
bold_dst=subj_dir / f"{SUBJ}_BOLD.nii.gz"
epr_src=ROOT.parent / "fmriprep_output" / SUBJ / "func" / \
    f"{SUBJ}_task-epr_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
assert epr_src.exists(), f"source epr BOLD missing: {epr_src}"
assert (subj_dir / f"{SUBJ}_Confounds.tsv").exists(), "Confounds.tsv missing"
assert (subj_dir / f"{SUBJ}_events.tsv").exists(), "events.tsv missing"
print(f"placing BOLD: {epr_src.name}\n           -> {bold_dst}")
shutil.copy2(epr_src, bold_dst)
print(f"placed sub-008_BOLD.nii.gz ({bold_dst.stat().st_size} bytes)")
cfc.get_included_subjects = lambda *a, **k: [SUBJ]
print("running preprocessBOLD.execute() scoped to sub-008 .....")
cfc.preprocessBOLD().execute()
print("execute() finished")
bold_dst.unlink()
print("removed temporary sub-008_BOLD.nii.gz (consistent with other subjects)")
