import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy

_REPO_ROOT=Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import config

class attention_interpreter():
    def __init__ (self,model,attention_pool,dataloader,device):
        self.model=model
        self.attention_pool=attention_pool
        self.dataloader=dataloader
        self.device=device
        model.eval()
        attention_pool.eval()
    def extract_node_attention(self):
        import torch
        self.model.eval()
        results={}
        with torch.no_grad():
            for i in self.dataloader:
                i=i.to(self.device) 
                output=self.model(i)
                for sub in range(len(i.subject_id)):              
                    a=i.subject_id[sub]
                    results[a]=output[sub].detach().cpu()
                
        return results


    def subtype_importance(self,attention_w,s_label):
        group={}
        for subject_id, label in s_label.items():
            if label not in group:
                group[label]=[]
            group[label].append(attention_w[subject_id])

        for cat in group:
            group[cat]=np.mean(group[cat],axis=0)
        return group
    
    def map_to_net(self, importance):
        from nilearn import datasets
        a=datasets.fetch_atlas_schaefer_2018(n_rois=config.nNodes)
        label=a['labels']
        networks={}
        parcel_idx=0
        for label_str in label:
            text=label_str.decode('utf-8') if isinstance(label_str,bytes) else label_str
            parts=text.split('_')
            if len(parts) < 3:
                continue
            network=parts[2]
            if network not in networks:
                networks[network]=[]
            networks[network].append(importance[parcel_idx])
            parcel_idx+=1
        if parcel_idx!=config.nNodes:
            return None,f"expected {config.nNodes} Schaefer parcels after dropping non-parcel labels, got {parcel_idx}"
        for i in networks:
            networks[i]=np.mean(networks[i],axis=0)
        return networks,None
    
    def get_condition_weights(self):
        import torch
        self.model.eval()
        aw=[]
        taus=[] 
        with torch.no_grad():
            for i in self.dataloader:
                i=i.to(self.device)
                embedd=self.model(i)
                pooled, weights, tau=self.attention_pool(embedd)  
                aw.append(weights)
                taus.append(tau)  
        mean_weights=torch.cat(aw).mean(dim=0) 
        weight_entropy=float(entropy(mean_weights.detach().cpu().numpy(), base=2))
        mean_tau=float(torch.stack(taus).mean().item())
        return {"mean_weights": mean_weights, "entropy": weight_entropy, "tau_attn": mean_tau}


#yeo-krienen 7-network mapping for per-region GATv2 attention
#real per-region attention depends on gnn_encoder.forward retrun val (not wired)
#the mapping and aggregation below are written against per-region attention arrays of the correct shape

# Yeo-Krienen 7 functional networks
YEO7_NETWORKS=("Visual","Somatomotor","Dorsal Attention","Ventral Attention","Limbic","Frontoparietal","Default Mode")

#schaefer-2018 (nilearn yeo_networks=7) parcel-label tokens to Yeo-7 names
#It is NOT an invented 200 to 7 mapping just implementation
SCHAEFER_TOKEN_TO_YEO7={
    "Vis":"Visual",
    "SomMot":"Somatomotor",
    "DorsAttn":"Dorsal Attention",
    "SalVentAttn":"Ventral Attention",
    "Limbic":"Limbic",
    "Cont":"Frontoparietal",
    "Default":"Default Mode",
}


def load_schaefer_yeo7_lookup(n_rois=None):
    #build the Schaefer-200 to Yeo-7 lookup: one network name per parcel
    n_rois=config.nNodes if n_rois is None else n_rois
    try:
        from nilearn import datasets
    except Exception as error:
        return None,f"nilearn unavailable, cannot source Schaefer-2018/Yeo-7 lookup: {type(error).__name__}: {error}"
    atlas=datasets.fetch_atlas_schaefer_2018(n_rois=n_rois,yeo_networks=7)
    region_networks=[]
    for raw in atlas["labels"]:
        text=raw.decode("utf-8") if isinstance(raw,bytes) else raw
        parts=text.split("_")
        if len(parts) < 3:
            continue
        token=parts[2]
        if token not in SCHAEFER_TOKEN_TO_YEO7:
            return None,f"unrecognized Schaefer parcel token {token!r} (expected one of {sorted(SCHAEFER_TOKEN_TO_YEO7)})"
        region_networks.append(SCHAEFER_TOKEN_TO_YEO7[token])
    if len(region_networks)!=n_rois:
        return None,f"expected {n_rois} Schaefer parcels after dropping non-parcel labels, got {len(region_networks)}"
    return np.array(region_networks),None


def aggregate_by_network(attention,region_networks):
    #collapse per-region attention into one mean value per Yeo-7 network
    attention=np.asarray(attention,dtype=float)
    if attention.ndim==2:
        attention=attention.mean(axis=0)
    region_networks=np.asarray(region_networks)
    if attention.shape[0]!=region_networks.shape[0]:
        raise ValueError(f"attention has {attention.shape[0]} regions but lookup has {region_networks.shape[0]}")
    per_network={}
    for net in YEO7_NETWORKS:
        mask=region_networks==net
        if mask.any():
            per_network[net]=float(attention[mask].mean())
    return per_network


def network_attention_by_subtype(subject_attention,subtype_labels,region_networks):
    #aggregate per-region attention into per-network means for each discovered FM subtype.
    grouped={}
    for subject_id,label in subtype_labels.items():
        if subject_id not in subject_attention:
            continue
        grouped.setdefault(label,[]).append(np.asarray(subject_attention[subject_id],dtype=float))
    rows={}
    for label,stack in grouped.items():
        subtype_mean=np.mean(np.stack(stack,axis=0),axis=0)
        rows[label]=aggregate_by_network(subtype_mean,region_networks)
    frame=pd.DataFrame(rows).T
    frame=frame.reindex(columns=[net for net in YEO7_NETWORKS if net in frame.columns])
    frame.index.name="subtype"
    return frame


def elevated_networks_per_subtype(network_frame,top_n=2):
    #for each subtype name the networks with the highest mean attention.
    report={}
    for subtype,row in network_frame.iterrows():
        ranked=row.sort_values(ascending=False)
        report[subtype]=list(ranked.index[:top_n])
    return report


def load_subtype_labels(csv_path=None):
    #read subject subtype assignments produced by src/clustering.py
    csv_path=config.clusterOutput/"K-Means-Labeling.csv" if csv_path is None else Path(csv_path)
    if not csv_path.exists():
        return None,f"no subtype labels found (expected {csv_path}); run src/clustering.py first"
    frame=pd.read_csv(csv_path)
    labels={row["Subject_Id"]:int(row["Label"]) for _,row in frame.iterrows()}
    return labels,None


def structural_self_test():
    #prove the mapping and aggregation are structurally sound against synthetic per-region attention
    rng=np.random.default_rng(config.randomSeed)

    print("[self-test] synthetic Schaefer-200 -> Yeo-7 lookup ...")
    #placehold for the nilearn lookup
    region_networks=np.array([YEO7_NETWORKS[i % len(YEO7_NETWORKS)] for i in range(config.nNodes)])
    assert region_networks.shape[0]==config.nNodes, "lookup must cover every parcel"
    assert set(region_networks)==set(YEO7_NETWORKS), "lookup must span all 7 Yeo networks"
    print(f"  parcels={region_networks.shape[0]} networks={len(set(region_networks))}")

    print("[self-test] single-subject aggregation ([N_NODES] -> 7) ...")
    one_subject=rng.random(config.nNodes)
    per_network=aggregate_by_network(one_subject,region_networks)
    assert len(per_network)==7, "expected 7 Yeo networks"
    print(f"  networks={list(per_network)}")

    print("[self-test] batched aggregation ([n_subjects, N_NODES] -> 7) ...")
    batched=rng.random((5,config.nNodes))
    per_network_batched=aggregate_by_network(batched,region_networks)
    assert len(per_network_batched)==7, "batched aggregation must still yield 7 networks"
    print(f"  batched networks={len(per_network_batched)}")

    print("[self-test] per-subtype grouping (fake K-Means labels) ...")
    n_subjects=12
    subject_attention={f"sub-{i:03d}":rng.random(config.nNodes) for i in range(n_subjects)}
    subtype_labels={f"sub-{i:03d}":(i % 2) for i in range(n_subjects)}
    frame=network_attention_by_subtype(subject_attention,subtype_labels,region_networks)
    assert frame.shape==(2,7), f"expected 2 subtypes x 7 networks, got {frame.shape}"
    print(frame.round(4).to_string())

    print("[self-test] elevated networks per subtype (Section 4.6 fill-in) ...")
    elevated=elevated_networks_per_subtype(frame,top_n=2)
    for subtype,nets in elevated.items():
        assert len(nets)==2, "each subtype should surface its top-2 networks"
        print(f"  subtype {subtype}: {', '.join(nets)}")
    print("[self-test] Good! Mapping + aggregation are structurally sound and ready to run.")
    return True


def main():
    print("=" * 70)
    print("Network attention mapping (paper Section 4.6)")
    print("  Schaefer-2018 200-parcel atlas -> Yeo-Krienen 7 networks")
    print("=" * 70)
    blockers=[]

    region_networks,lookup_blocker=load_schaefer_yeo7_lookup()
    if region_networks is None:
        blockers.append(f"Schaefer->Yeo7 lookup BLOCKED: {lookup_blocker}")
    else:
        print(f"[lookup] {region_networks.shape[0]} parcels mapped to {len(set(region_networks))} Yeo-7 networks")

    subtype_labels,label_blocker=load_subtype_labels()
    if subtype_labels is None:
        blockers.append(f"subtype labels BLOCKED: {label_blocker}")

    #real per-region attention depends on gnn_encoder.forward(return_attention_weights=...)
    #which isnt wired yet, so live per-network aggregation is put off for now
    blockers.append("real per-region attention BLOCKED: gnn_encoder return_attention_weights not wired yet")

    if blockers:
        print("\n" + "-" * 70)
        print("BLOCKERS (no results fabricated):")
        for b in blockers:
            print(f"  - {b}")
        print("-" * 70)
        print("Running structural self-test against synthetic data instead ...")
        structural_self_test()
    return blockers


if __name__ == "__main__":
    main()
