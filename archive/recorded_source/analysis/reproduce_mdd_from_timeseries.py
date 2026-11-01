"""Rebuild the depression-cohort embedding from parcel time series with the saved canonical checkpoint.

The inputs are the 72 OpenNeuro ds002748 parcel time series in ../resting_state_dep_data, the canonical
checkpoint and its selected hyperparameters. The fMRIPrep confound files used for the motion filter are not
available, so the included patients are taken from the saved K-Means-Labeling.csv. Normalization uses the
statistics stored in the checkpoint, so the result does not depend on which controls passed the filter.
"""
import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(v,'1')
import argparse,hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score
R=Path(__file__).resolve().parents[1]
for p in (R,R/'src',R/'models',R/'experiments/ds002748_mdd'):
    sys.path.insert(0,str(p))
from config import config
import dataset_mdd
C=R/'results/mdd/canonical_single'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path,default=R/'reproducibility/audits/mdd_timeseries_reproduction.json')
    a=a.parse_args()
    params=json.loads((C/'bestParamsMdd.json').read_text())
    for k in ('dModel','heads','output','layers'):
        setattr(config,k,int(params[k]['0']))
    for k in ('dropout','lr','weightDecay','maskRate','ntXentTemp'):
        setattr(config,k,float(params[k]['0']))
    labels=pd.read_csv(C/'K-Means-Labeling.csv')
    ids=labels['Subject_Id'].astype(str).tolist()
    dataset_mdd.get_included_subjects_mdd=lambda:sorted(ids)
    ds=dataset_mdd.mddDataset()
    ck=torch.load(C/'bestMddModel.pt',map_location='cpu',weights_only=False)
    ds.applyNormalization(ck['mu'],ck['sig'])
    from gnn_encoder import GNNEncoder
    enc=GNNEncoder()
    enc.load_state_dict(ck['enc'])
    enc.eval()
    by={s['subject_id']:s for s in ds.subjectData}
    assert set(by)==set(ids),'Missing time series for included patients'
    assert all(int(by[s]['group_label'].item())==0 for s in ids),'Included subject is not a patient'
    with torch.no_grad():
        emb=np.stack([enc(Batch.from_data_list(by[s]['graphs'])).view(-1).numpy() for s in ids])
    saved=np.load(C/'Embeddings.npy')
    diff=float(np.max(np.abs(emb-saved)))
    x=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-08)
    ref=pd.read_csv(C/'silhouette-scores.csv')
    sil={}
    part=None
    for k in config.kmeansKRange:
        ls=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=config.randomSeed).fit_predict(x)
        sil[k]=float(silhouette_score(x,ls))
        if k==int(ref['k_selected_silhouette'].iloc[0]):
            part=ls
    saved_sil={int(k):float(v) for k,v in zip(ref['k'],ref['silhouette_score'])}
    out=dict(scope='Parcel time series to embedding, k-means partition and silhouettes for the 51 depression patients, using the saved canonical checkpoint. Upstream fMRIPrep preprocessing and training are not rerun.',
        n_patients=len(ids),embedding_shape=list(emb.shape),max_abs_embedding_difference=diff,embeddings_match_1e_5=diff<1e-5,
        silhouettes_recomputed=sil,silhouettes_saved=saved_sil,max_abs_silhouette_difference=max(abs(sil[k]-saved_sil[k]) for k in sil),
        selected_k=int(ref['k_selected_silhouette'].iloc[0]),partition_ari_vs_saved=float(adjusted_rand_score(labels['Label'],part)),
        inputs=dict(checkpoint_sha256=sha(C/'bestMddModel.pt'),params_sha256=sha(C/'bestParamsMdd.json'),saved_embedding_sha256=sha(C/'Embeddings.npy'),timeseries_sha256={s:sha(dataset_mdd.MDD_ROOT/dataset_mdd.TS_PAT.format(s=s)) for s in ids},participants_tsv_sha256=sha(dataset_mdd.PART_TSV)),
        source_sha256={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in [Path(__file__),R/'experiments/ds002748_mdd/dataset_mdd.py',R/'models/gnn_encoder.py',R/'config.py']},
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))
    args=a
    args.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:out[k] for k in ('n_patients','max_abs_embedding_difference','embeddings_match_1e_5','max_abs_silhouette_difference','partition_ari_vs_saved')},indent=2))

if __name__=='__main__':
    main()
