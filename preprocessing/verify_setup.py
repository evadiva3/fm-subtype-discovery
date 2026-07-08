import sys;
import numpy as np;
import pandas as pd;
from pathlib import Path;

sys.path.insert(0, str(Path(__file__).resolve().parent.parent));
from config import config;

DATA_FOLDER=config.DATA_ROOT;

TR=config.TR;
FD_THRESHOLD=0.5;
FRACTION_THRESHOLD=0.25;
FD_COLUMN="framewise_displacement";

CONDITIONS=config.CONDITIONS;


def _clean(name):
    return name.replace(" ", "").replace("-", "");


def load_fd(confounds_path):
    df=pd.read_csv(confounds_path, sep="\t");
    if FD_COLUMN not in df.columns:
        raise ValueError(f"missing '{FD_COLUMN}' column in {confounds_path.name}");
    fd=df[FD_COLUMN].to_numpy(dtype=float).copy();
    if fd.size and np.isnan(fd[0]):
        fd[0]=0.0;
    return fd;


def slice_conditions(fd, events_path):
    events=pd.read_csv(events_path, sep="\t");
    for col in ("trial_type", "onset", "duration"):
        if col not in events.columns:
            raise ValueError(f"missing '{col}' column in {events_path.name}");
    per_cond={c: [] for c in CONDITIONS};
    for row in events.itertuples(index=False):
        trial_type=getattr(row, "trial_type");
        if trial_type in per_cond:
            start_tr=int(row.onset / TR);
            end_tr=int((row.onset + row.duration) / TR);
            per_cond[trial_type].append(fd[start_tr:end_tr]);
    return {c: (np.concatenate(s) if s else np.array([], dtype=float)) for c, s in per_cond.items()};


def evaluate_subject(subject_dir):
    sid=subject_dir.name;
    result={"subject_id": sid, "excluded": False, "frac_above_threshold": np.nan, "error": ""};
    for c in CONDITIONS:
        result[f"meanFD_{_clean(c)}"]=np.nan;
    try:
        confounds_path=subject_dir / f"{sid}_Confounds.tsv";
        events_path=subject_dir / f"{sid}_events.tsv";
        if not confounds_path.exists():
            raise FileNotFoundError(f"{confounds_path.name} not found");
        if not events_path.exists():
            raise FileNotFoundError(f"{events_path.name} not found");

        fd=load_fd(confounds_path);
        cond_fd=slice_conditions(fd, events_path);

        for c in CONDITIONS:
            arr=cond_fd[c];
            result[f"meanFD_{_clean(c)}"]=float(np.mean(arr)) if arr.size else np.nan;

        pooled=np.concatenate([cond_fd[c] for c in CONDITIONS]);
        if pooled.size == 0:
            raise ValueError("no FD volumes matched any condition epoch");
        frac=float((pooled > FD_THRESHOLD).sum() / pooled.size);
        result["frac_above_threshold"]=frac;
        result["excluded"]=bool(frac > FRACTION_THRESHOLD);
    except Exception as error:
        result["excluded"]=True;
        result["error"]=f"{type(error).__name__}: {error}";
    return result;


def apply_exclusions(excluded_ids, data_folder):
    # Move excluded subject folders into data_folder/excluded/ so downstream
    # pipeline steps (preprocessBOLD.execute(), dataset.py) simply won't see them
    excluded_dir=data_folder / "excluded";
    excluded_dir.mkdir(exist_ok=True);

    log_rows=[];
    for sid in excluded_ids:
        src=data_folder / sid;
        dst=excluded_dir / sid;
        if dst.exists():
            print(f"{sid} already in excluded/, skipping move.");
            log_rows.append({"subject_id": sid, "action": "already_excluded", "timestamp": pd.Timestamp.now().isoformat()});
            continue;
        if src.exists():
            src.rename(dst);
            print(f"Moved {sid} -> {dst}");
            log_rows.append({"subject_id": sid, "action": "moved", "timestamp": pd.Timestamp.now().isoformat()});
        else:
            print(f"{sid} flagged for exclusion but not found at {src} (already moved elsewhere, or never existed).");
            log_rows.append({"subject_id": sid, "action": "not_found", "timestamp": pd.Timestamp.now().isoformat()});

    log_path=excluded_dir / "move_log.csv";
    log_df=pd.DataFrame(log_rows);
    if log_path.exists():
        existing=pd.read_csv(log_path);
        log_df=pd.concat([existing, log_df], ignore_index=True);
    log_df.to_csv(log_path, index=False);
    print(f"Move log written to {log_path}");


def main():
    if not DATA_FOLDER.exists():
        print(f"DATA_FOLDER does not exist: {DATA_FOLDER.resolve()}");
        print("Set DATA_FOLDER to the same path preprocessBOLD.py uses, then re-run.");
        return;

    subject_dirs=sorted(
        d for d in DATA_FOLDER.iterdir()
        if d.is_dir() and not d.name.startswith("top_") and d.name != "excluded"
    );
    results=[evaluate_subject(d) for d in subject_dirs];

    print(f"{'subject_id':<20} {'excluded':<9} {'frac>0.5mm':<11} error");
    print("-" * 64);
    for r in results:
        frac=r["frac_above_threshold"];
        frac_s="nan" if (frac is None or np.isnan(frac)) else f"{frac:.3f}";
        print(f"{r['subject_id']:<20} {str(r['excluded']):<9} {frac_s:<11} {r['error']}");
    excluded_ids=[r["subject_id"] for r in results if r["excluded"]];
    print(f"\n{len(excluded_ids)} of {len(results)} subject(s) flagged for exclusion.");

    if results:
        cond_cols=[f"meanFD_{_clean(c)}" for c in CONDITIONS];
        ordered=["subject_id", "excluded", "frac_above_threshold", "error"] + cond_cols;
        pd.DataFrame(results)[ordered].to_csv(DATA_FOLDER / "fd_check_results.csv", index=False);
    with open(DATA_FOLDER / "motion_excluded_subjects.txt", "w") as f:
        for sid in excluded_ids:
            f.write(sid + "\n");
    print(f"Wrote {DATA_FOLDER / 'fd_check_results.csv'} and {DATA_FOLDER / 'motion_excluded_subjects.txt'}");

    apply_exclusions(excluded_ids, DATA_FOLDER);


if __name__ == "__main__":
    main();