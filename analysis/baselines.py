import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from nilearn.decomposition import CanICA
from nilearn.image import concat_imgs
from config import config
class baselines:
    def __init__(self,fc_matrices,labels,preprocessed_fmri_dir=None,subject_ids=None):
        labels=np.asarray(labels);fc_matrices=np.asarray(fc_matrices);self._fc_matrices_all=fc_matrices;self._labels_all=labels;f=labels==0
        if not f.any():raise ValueError("baselines: no FM subjects (label==0)can be found; FM-only baselines cannot run")
        self.fc_matrices=fc_matrices[f];self.labels=labels[f];self.preprocessed_fmri_dir=preprocessed_fmri_dir
        if subject_ids is not None:self._fm_subject_ids=[s for s,k in zip(np.asarray(subject_ids),f) if k]
        else:self._fm_subject_ids=None
    @staticmethod
    def _l2(M):
        M=np.asarray(M,dtype=float);return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-08)
    def pca_kmeans(self,k,n_comp=config.pcaComponents):
        f=self.fc_matrices.reshape(len(self.fc_matrices),-1);ns,n=f.shape;e=min(n_comp,ns,n);p=PCA(n_components=e);r=self._l2(p.fit_transform(f))
        km=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=config.randomSeed);a=km.fit(r);l=a.labels_;s=silhouette_score(r,l);return(l,s,r)
    def flat_triangle_kmeans(self,k):
        n=self.fc_matrices.shape[-1];r,c=np.triu_indices(n,k=1);f=np.array([m[r,c] for m in self.fc_matrices]);e=n*(n-1)//2;assert f.shape[1]==e,f"expected {e}-dim vectors, got {f.shape[1]}"
        f=self._l2(f);km=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=config.randomSeed);a=km.fit(f);l=a.labels_;s=silhouette_score(f,l);return(l,s,f)
    def _discover_subject_niftis(self,base,sid):
        s="space-MNI152NLin2009cAsym";t="task-epr";ps=[f"{sid}/func/*{t}*{s}*desc-preproc_bold.nii*",f"{sid}/ses-*/func/*{t}*{s}*desc-preproc_bold.nii*",f"{sid}/**/*{t}*{s}*desc-preproc_bold.nii*"]
        for p in ps:
            c=sorted(base.glob(p))
            if c:return c
        return[]
    def group_ica_kmeans(self):
        if self.preprocessed_fmri_dir is None:raise ValueError("group_ica_kmeans requires preprocessed_fmri_dir (dir of 4D BOLD niftis) at init")
        if self._fm_subject_ids is None:raise ValueError("group_ica_kmeans requires subject_ids at init to locate each FM subject's niftis")
        b=Path(self.preprocessed_fmri_dir);ss=[];ks=[]
        for sd in self._fm_subject_ids:
            n=self._discover_subject_niftis(b,sd)
            if not n:
                print(f"[group_ica_kmeans] warning: skipping subject {sd} , no preprocessed niftis found under {b}");continue
            sn=concat_imgs(n) if len(n)>1 else n[0];ss.append(sn);ks.append(sd)
        if len(ss)<2:raise ValueError(f"group_ica_kmeans: only {len(ss)} FM subject had usable niftis; need >=2 to cluster")
        c=CanICA(n_components=config.groupIcaComponents,mask_strategy="whole-brain-template",random_state=config.randomSeed);c.fit(ss);f=[]
        for i in ss:
            t=c.transform([i])[0];f.append(np.abs(t).mean(axis=0))
        f=self._l2(np.array(f));bl,be,bk=(None,-1.0,None)
        for k in(2,3,4):
            km=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=config.randomSeed);l=km.fit_predict(f);s=silhouette_score(f,l)
            if s>be:bl,be,bk=(l,s,k)
        return(bl,be,bk,f)
    def svm_classification(self):
        f=self._fc_matrices_all.reshape(len(self._fc_matrices_all),-1);s=SVC();cv=cross_val_score(s,f,self._labels_all,cv=config.svmCvFolds);return cv.mean()
