import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
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
        mu,sd,obs,z,pc=0.192,0.028,0.150,-1.51,5.2
        rng=np.random.default_rng(config.randomSeed)
        d=rng.normal(mu,sd,1000)
        fig,ax=plt.subplots(figsize=(7,5))
        ax.hist(d,bins=40,density=True,color='pink',edgecolor='white')
        ax.axvline(obs,color='crimson',lw=2)
        ax.annotate(f"z = {z}, exceeds only {pc}% of null draws",xy=(obs,ax.get_ylim()[1]*0.9),xytext=(mu+sd,ax.get_ylim()[1]*0.9),arrowprops=dict(arrowstyle='->',color='crimson'),va='center')
        ax.set_title("FM-only k=3 clustering: observed silhouette vs. covariance-preserving null (n=1000)",fontsize=9)
        ax.set_xlabel("silhouette score")
        ax.set_ylabel("density")
        self._sv(fig,'fig1_null_silhouette')
    def plot_ablation(self):
        nm=['Full model','Mean pooling','Untrained encoder']
        sil=[0.168,0.240,0.520]
        lo=[0,0,0.520-0.43]
        hi=[0,0,0]
        sd=[0.3811532576634764,0.3433691061855307,0.3339335233576474,0.32217302996386665,0.3932047322574061,0.08614835399249417,0.41169575718497503,0.3346713696098955,0.3217275400177396,0.2759226691725773]
        mn,ot=-0.044572519455699504,-0.2230989402720051
        fig,ax=plt.subplots(1,2,figsize=(12,5))
        ax[0].bar(nm,sil,yerr=[lo,hi],capsize=6,color='pink',edgecolor='white')
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
        k=[2,3,4,5,6]
        g=[0.412,0.532,0.622,0.695,0.770]
        fig,ax=plt.subplots(figsize=(7,5))
        ax.plot(k,g,marker='o',color='crimson')
        ax.set_xticks(k)
        ax.set_xlabel("k")
        ax.set_ylabel("gap statistic")
        ax.set_title("Gap statistic across k")
        ax.annotate("no interior peak; 1-SE rule returns range ceiling (k=6), not a detected peak",xy=(k[-1],g[-1]),xytext=(k[0]+0.1,g[-1]),fontsize=8,va='top')
        self._sv(fig,'fig3_gap_statistic')


