#Compare a fresh fixed-architecture retraining run with the saved run for the same seed.
#Run experiments/ds002748_mdd/propagated_null/mdd_fixed_arch_control.py with MDD_CTRL_ROOT set to a fresh
#directory first. The saved runs are results/mdd/fixed_arch_control/runs/seed###.json and emb###.npy.
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
R=Path(__file__).resolve().parents[1]
S=R/"results/mdd/fixed_arch_control/runs"

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--fresh",type=Path,required=True,help="MDD_CTRL_ROOT used for the fresh run")
    a.add_argument("--seed",type=int,default=0)
    a.add_argument("--environment",default="")
    a.add_argument("--output",type=Path,default=R/"reproducibility/audits/mdd_retraining_reproduction.json")
    a=a.parse_args()
    n=f"{a.seed:03d}"
    old=json.loads((S/f"seed{n}.json").read_text())
    new=json.loads((a.fresh/"runs"/f"seed{n}.json").read_text())
    eo=np.load(S/f"emb{n}.npy");en=np.load(a.fresh/"runs"/f"emb{n}.npy")
    assert old["subject_ids"]==new["subject_ids"]
    diff=float(np.max(np.abs(eo-en)))
    out=dict(scope="One fixed-architecture retraining seed rerun from the ds002748 parcel time series and compared with the saved run.",
        seed=a.seed,n_patients=len(old["subject_ids"]),bitwise_match=diff==0.,max_abs_embedding_difference=diff,
        selected_k_saved=old["k_selected"],selected_k_retrained=new["k_selected"],silhouette_saved=old["silhouette"],silhouette_retrained=new["silhouette"],
        partition_ari=float(adjusted_rand_score(old["labels"],new["labels"])),elapsed_s_retrained=new.get("elapsed_s"),environment=a.environment,
        saved_embedding_sha256=hashlib.sha256((S/f"emb{n}.npy").read_bytes()).hexdigest(),retrained_embedding_sha256=hashlib.sha256((a.fresh/"runs"/f"emb{n}.npy").read_bytes()).hexdigest())
    a.output.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
