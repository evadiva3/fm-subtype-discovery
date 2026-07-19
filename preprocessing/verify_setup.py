import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
DATA_FOLDER=config.subjectDataFolder
TR=config.tr
FD_THRESHOLD=config.fdThreshold
FRACTION_THRESHOLD=config.fdFractionThreshold
FD_COLUMN="framewise_displacement"
CONDITIONS=config.conditions
CANON=re.compile(r"^sub-\d+$") 
KNOWN_MISSING={"sub-005", "sub-012", "sub-032"} 


def _clean(name):
    return name.replace(" ", "").replace("-", "")


def load_fd(confounds_path):
    df=pd.read_csv(confounds_path, sep="\t")
    if FD_COLUMN not in df.columns:
        raise ValueError(f"missing '{FD_COLUMN}' column in {confounds_path.name}")
    fd=df[FD_COLUMN].to_numpy(dtype=float).copy()
    if fd.size and np.isnan(fd[0]):
        fd[0]=0.0
    return fd


def slice_conditions(fd, events_path):
    events=pd.read_csv(events_path, sep="\t")
    for col in ("trial_type", "onset", "duration"):
        if col not in events.columns:
            raise ValueError(f"missing '{col}' column in {events_path.name}")
    per_cond={c: [] for c in CONDITIONS}
    for row in events.itertuples(index=False):
        trial_type=getattr(row, "trial_type")
        if trial_type in per_cond:
            start_tr=int(row.onset / TR)
            end_tr=int((row.onset + row.duration) / TR)
            per_cond[trial_type].append(fd[start_tr:end_tr])
    return {c: (np.concatenate(s) if s else np.array([], dtype=float)) for c, s in per_cond.items()}


def _reason(r):
    if not CANON.match(r["subject_id"]):
        return "junk_duplicate"
    if not r["excluded"]:
        return ""
    return "missing_file" if r["error"] else "motion"

def evaluate_subject(subject_dir):
    sid=subject_dir.name
    result={"subject_id": sid, "excluded": False, "exclusion_reason": "", "frac_above_threshold": np.nan, "error": ""}
    for c in CONDITIONS:
        result[f"meanFD_{_clean(c)}"]=np.nan
    try:
        confounds_path=subject_dir / f"{sid}_Confounds.tsv"
        events_path=subject_dir / f"{sid}_events.tsv"
        if not confounds_path.exists():
            raise FileNotFoundError(f"{confounds_path.name} not found")
        if not events_path.exists():
            raise FileNotFoundError(f"{events_path.name} not found")

        fd=load_fd(confounds_path)
        cond_fd=slice_conditions(fd, events_path)

        for c in CONDITIONS:
            arr=cond_fd[c]
            result[f"meanFD_{_clean(c)}"]=float(np.mean(arr)) if arr.size else np.nan

        pooled=np.concatenate([cond_fd[c] for c in CONDITIONS])
        if pooled.size == 0:
            raise ValueError("no FD volumes matched any condition epoch")
        frac=float((pooled > FD_THRESHOLD).sum() / pooled.size)
        result["frac_above_threshold"]=frac
        result["excluded"]=bool(frac > FRACTION_THRESHOLD)
    except Exception as error:
        result["excluded"]=True
        result["error"]=f"{type(error).__name__}: {error}"
    if not CANON.match(sid):
        result["excluded"]=True
    result["exclusion_reason"]=_reason(result)
    return result


def _relocate(ids, data_folder, dest, log_name):
    if not ids:
        return
    dd=data_folder / dest
    dd.mkdir(exist_ok=True)
    rows=[]
    for sid in ids:
        src=data_folder / sid
        dst=dd / sid
        if dst.exists():
            rows.append({"subject_id": sid, "action": "already_there", "timestamp": pd.Timestamp.now().isoformat()})
        elif src.exists():
            src.rename(dst)
            print(f"  {sid} -> {dest}/")
            rows.append({"subject_id": sid, "action": "moved", "timestamp": pd.Timestamp.now().isoformat()})
        else:
            rows.append({"subject_id": sid, "action": "not_found", "timestamp": pd.Timestamp.now().isoformat()})
    log=dd / log_name
    df=pd.DataFrame(rows)
    if log.exists() and log.stat().st_size>0:
        df=pd.concat([pd.read_csv(log), df], ignore_index=True)
    df.to_csv(log, index=False)
    print(f"  log -> {log}")


def main():
    if not DATA_FOLDER.exists():
        print(f"DATA_FOLDER does not exist: {DATA_FOLDER.resolve()}")
        print("Set DATA_FOLDER to the same path preprocessBOLD.py uses, then re-run.")
        return

    skip={"excluded", "junk_removed"}
    subject_dirs=sorted(
        d for d in DATA_FOLDER.iterdir()
        if d.is_dir() and not d.name.startswith("top_") and d.name not in skip
    )
    results=[evaluate_subject(d) for d in subject_dirs]
    junk=[r["subject_id"] for r in results if r["exclusion_reason"]=="junk_duplicate"]
    miss=[r["subject_id"] for r in results if r["exclusion_reason"]=="missing_file"]
    mot=[r["subject_id"] for r in results if r["exclusion_reason"]=="motion"]
    clean=[r["subject_id"] for r in results if r["exclusion_reason"]==""]

    if results:
        cond_cols=[f"meanFD_{_clean(c)}" for c in CONDITIONS]
        ordered=["subject_id", "excluded", "exclusion_reason", "frac_above_threshold", "error"] + cond_cols
        mp=config.exclusionManifestPath
        mp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results)[ordered].to_csv(mp, index=False)
        print(f"\nManifest -> {mp}")

    print(f"Removing {len(junk)} junk duplicate(s) -> junk_removed/")
    _relocate(junk, DATA_FOLDER, "junk_removed", "junk_log.csv")
    print(f"Keeping {len(mot)+len(miss)} real excluded subject(s) on disk (manifest-flagged, not moved)")

    # summary
    print("\n" + "=" * 60)
    print("FINAL BREAKDOWN")
    print("=" * 60)
    print(f"  scanned folders          : {len(results)}")
    print(f"  junk duplicates removed  : {len(junk)}")
    print(f"  missing_file (known excl): {len(miss)}  {miss}")
    known_seen=sorted(KNOWN_MISSING & set(miss))
    known_absent=sorted(KNOWN_MISSING - set(miss))
    print(f"    reconcile vs known {sorted(KNOWN_MISSING)}: seen={known_seen} absent(not a folder)={known_absent}")
    print(f"  real motion exclusions   : {len(mot)}  {mot}")
    print(f"  final clean subjects     : {len(clean)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
