#simple model to prove complexity=good *supervised*
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score

class baselines():
    def __init__(self, fc_matrices,labels):
        self.fc_matrices=fc_matrices
        self.labels=labels
    def pca_kmeans(self,k,n_comp=50):
        flat=self.fc_matrices.reshape(len(self.fc_matrices),-1)
        pca=PCA(n_components=n_comp)
        reduce= pca.fit_transform(flat)
        km= KMeans(n_clusters=k)
        a=km.fit(reduce)
        label=a.labels_
        score=silhouette_score(reduce,label)
        return(label,score)
    def raw_kmean(self,k):
        flat=self.fc_matrices.reshape(len(self.fc_matrices),-1)
        km= KMeans(n_clusters=k)
        a=km.fit(flat)
        label=a.labels_
        score=silhouette_score(flat,label)
        return(label,score)
    def svm_classification(self):
        flat=self.fc_matrices.reshape(len(self.fc_matrices),-1)
        svm=SVC()
        cv=cross_val_score(svm,flat,self.labels,cv=5)
        return cv.mean()