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

class baselines():
    def __init__(self, fc_matrices, labels, preprocessed_fmri_dir=None, subject_ids=None): 
        labels=np.asarray(labels)  
        fc_matrices=np.asarray(fc_matrices)  
        self._fc_matrices_all=fc_matrices  
        self._labels_all=labels  
        fm_mask=(labels==0)  
        if not fm_mask.any():  
            raise ValueError("baselines: no FM subjects (label==0)can be found; FM-only baselines cannot run")  
        self.fc_matrices=fc_matrices[fm_mask]  
        self.labels=labels[fm_mask]  
        self.preprocessed_fmri_dir=preprocessed_fmri_dir  
        if subject_ids is not None:  
            self._fm_subject_ids=[sid for sid, keep in zip(np.asarray(subject_ids), fm_mask) if keep]  
        else:
            self._fm_subject_ids=None 

    def pca_kmeans(self,k,n_comp=config.PCA_COMPONENTS):
        flat=self.fc_matrices.reshape(len(self.fc_matrices),-1)
        n_samples, n_features=flat.shape
        effective_components=min(n_comp, n_samples, n_features)
        pca=PCA(n_components=effective_components)
        reduce=pca.fit_transform(flat)
        km=KMeans(n_clusters=k, random_state=config.RANDOM_SEED)
        a=km.fit(reduce)
        label=a.labels_
        score=silhouette_score(reduce,label)
        return(label,score)

    def flat_triangle_kmeans(self,k):  
        #19,900 dim upper triangle-only flatten excludes diagonal and redundant mirrored lower triangle
        n=self.fc_matrices.shape[-1]  
        rows, cols=np.triu_indices(n, k=1) 
        flat=np.array([m[rows, cols] for m in self.fc_matrices])  
        expected_dim=n*(n-1)//2  
        assert flat.shape[1]==expected_dim, f"expected {expected_dim}-dim vectors, got {flat.shape[1]}"
        km=KMeans(n_clusters=k, random_state=config.RANDOM_SEED)
        a=km.fit(flat)
        label=a.labels_
        score=silhouette_score(flat,label)  
        return(label,score)

    def _discover_subject_niftis(self, base, sid):
        space="space-MNI152NLin2009cAsym"
        patterns=[
            f"{sid}/func/*{space}*desc-preproc_bold.nii*",      
            f"{sid}/ses-*/func/*{space}*desc-preproc_bold.nii*",
            f"{sid}/**/*{space}*desc-preproc_bold.nii*",     
        ]
        for pat in patterns: 
            candidates=sorted(base.glob(pat))
            if candidates:
                return candidates
        return [] 

    def group_ica_kmeans(self): 
        if self.preprocessed_fmri_dir is None:
            raise ValueError("group_ica_kmeans requires preprocessed_fmri_dir (dir of 4D BOLD niftis) at init")  
        if self._fm_subject_ids is None:  
            raise ValueError("group_ica_kmeans requires subject_ids at init to locate each FM subject's niftis") 
        base=Path(self.preprocessed_fmri_dir) 
        subject_imgs=[] 
        kept_ids=[]
        for sid in self._fm_subject_ids: 
            niftis=self._discover_subject_niftis(base, sid) 
            if not niftis: 
                print(f"[group_ica_kmeans] warning: skipping subject {sid} , no preprocessed niftis found under {base}") 
                continue 
            session=concat_imgs(niftis) if len(niftis)>1 else niftis[0]  
            subject_imgs.append(session)  
            kept_ids.append(sid) 
        if len(subject_imgs)<2: 
            raise ValueError(f"group_ica_kmeans: only {len(subject_imgs)} FM subject had usable niftis; need >=2 to cluster")  
        canica=CanICA(n_components=config.GROUP_ICA_COMPONENTS, mask_strategy='whole-brain-template', random_state=config.RANDOM_SEED)
        canica.fit(subject_imgs)
        features=[] 
        for img in subject_imgs: 
            time_courses=canica.transform([img])[0]
            features.append(np.abs(time_courses).mean(axis=0))
        features=np.array(features)
        best_label, best_score, best_k=None, -1.0, None 
        for k in (2, 3, 4):
            km=KMeans(n_clusters=k, random_state=config.RANDOM_SEED)
            label=km.fit_predict(features)
            score=silhouette_score(features, label) 
            if score>best_score: 
                best_label, best_score, best_k=label, score, k  
        return(best_label, best_score, best_k) 

    def svm_classification(self):
        flat=self._fc_matrices_all.reshape(len(self._fc_matrices_all),-1) 
        svm=SVC()
        cv=cross_val_score(svm,flat,self._labels_all,cv=config.SVM_CV_FOLDS)
        return cv.mean()
