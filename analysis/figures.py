import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import config

class figure_gen():
    def __init__(self,dire=None):
        self.dire=config.figuresDir if dire is None else dire
        os.makedirs(self.dire,exist_ok=True)
    def plot_loss(self,train_loss,val_loss):
        plt.plot(range(len(train_loss)), train_loss, label='train')
        plt.plot(range(len(val_loss)), val_loss, label='val')
        plt.savefig(os.path.join(self.dire, 'loss_plot.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def plot_silh_comp(self,methods,scores):
        sns.barplot(x=methods,y=scores,color='pink')
        plt.savefig(os.path.join(self.dire, 'silhuotte_comparision_plot.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def plot_clinical_violins(self,df,subtypes):
        sns.violinplot(data=df,x=subtypes,color='pink')  
        plt.savefig(os.path.join(self.dire, 'subtype_violin_plot.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def plot_attention_brain_map(self,weights,regions):
        sns.barplot(x=weights,y=regions,color='pink')
        plt.savefig(os.path.join(self.dire, 'attention_plot.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def plot_con_contri(self,attention_weight):
        conditions=config.conditions
        sns.barplot(x=(conditions),y=attention_weight,color='pink')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.dire, 'condition_contributions_plot.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def _sv(self,fig,nm):
        fig.savefig(os.path.join(self.dire,nm+'.png'),dpi=300,bbox_inches='tight')
        fig.savefig(os.path.join(self.dire,nm+'.svg'),bbox_inches='tight')
        plt.close(fig)
    def plot_null_silh(self):
        E=np.load(config.clusterOutput/"Embeddings.npy")
        lab=pd.read_csv(config.clusterOutput/"K-Means-Labeling.csv")["Label"].to_numpy()
        En=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-8)
        k=len(np.unique(lab))
        obs=silhouette_score(En,lab)
        mu=En.mean(0); Xc=En-mu; sc=np.sqrt(max(len(En)-1,1))
        rng=np.random.default_rng(config.randomSeed)
        d=np.empty(config.nPermutations)
        for i in range(config.nPermutations):
            nul=mu+(rng.standard_normal((len(En),len(En)))@Xc)/sc
            nl=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=i).fit_predict(nul)
            d[i]=silhouette_score(nul,nl)
        pc=float((d>obs).mean()*100)
        fig,ax=plt.subplots(figsize=(7,5))
        ax.hist(d,bins=40,density=True,color='pink',edgecolor='white')
        ax.axvline(obs,color='crimson',lw=2)
        ax.annotate(f"observed={obs:.3f}, exceeds {100-pc:.1f}% of null draws (p={pc/100:.3f})",xy=(obs,ax.get_ylim()[1]*0.9),xytext=(d.mean(),ax.get_ylim()[1]*0.9),arrowprops=dict(arrowstyle='->',color='crimson'),va='center',fontsize=8)
        ax.set_title(f"FM-only k={k} clustering: observed silhouette vs. covariance-preserving null (n={config.nPermutations})",fontsize=9)
        ax.set_xlabel("silhouette score")
        ax.set_ylabel("density")
        self._sv(fig,'fig1_null_silhouette')
    def plot_ablation(self):
        ab=pd.read_csv(config.resultsRoot/"ablation_table.csv").set_index("condition")
        sd=pd.read_csv(config.resultsRoot/"severity_gradient_random_seeds.csv")["r"].tolist()
        reg=pd.read_csv(config.resultsRoot/"severity_gradient_regression.csv").set_index("space")["r"]
        nm=['Full model','Mean pooling','Untrained encoder']
        sil=[float(ab.loc[n,"silhouette"]) for n in nm]
        mn=float(reg.loc["main-trained"]); ot=float(reg.loc["orthogonal-trained"])
        fig,ax=plt.subplots(1,2,figsize=(12,5))
        ax[0].bar(nm,sil,capsize=6,color='pink',edgecolor='white')
        ax[0].set_ylabel("silhouette score")
        ax[0].set_title("A. Ablation")
        sns.stripplot(x=[0]*len(sd),y=sd,ax=ax[1],color='pink',size=8,jitter=0.15)
        ax[1].axhline(mn,color='crimson',ls='--',label=f"trained main r={mn:.3f}")
        ax[1].axhline(ot,color='navy',ls='--',label=f"trained orthogonal r={ot:.3f}")
        ax[1].set_ylabel("r")
        ax[1].set_xticks([])
        ax[1].set_title("B. Random-encoder regression seeds (n=10)")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        self._sv(fig,'fig2_ablation')
    def plot_gap(self):
        sc=pd.read_csv(config.clusterOutput/"silhouette-scores.csv")
        k=sc["k"].tolist()
        g=sc["gap_stat"].tolist()
        kg=int(sc["k_selected_gap"].iloc[0])
        fig,ax=plt.subplots(figsize=(7,5))
        ax.plot(k,g,marker='o',color='crimson')
        ax.set_xticks(k)
        ax.set_xlabel("k")
        ax.set_ylabel("gap statistic")
        ax.set_title("Gap statistic across k")
        ax.annotate(f"no interior peak; 1-SE rule returns range ceiling (k={kg}), not a detected peak",xy=(k[-1],g[-1]),xytext=(k[0]+0.1,g[-1]),fontsize=8,va='top')
        self._sv(fig,'fig3_gap_statistic')


