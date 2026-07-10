#who is included reads the exclusion manifest returns kept subjects
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config


def get_included_subjects(manifest_path=None):
    path=config.exclusionManifestPath if manifest_path is None else Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"exclusion manifest not found at {path}; run preprocessing/verify_setup.py first")
    df=pd.read_csv(path)
    reason=df["exclusion_reason"].fillna("").astype(str)
    keep=df[reason==""]
    return sorted(keep["subject_id"].tolist())
