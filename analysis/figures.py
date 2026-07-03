import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

class figure_gen():
    def __init__(self,dire):
        self.dire=dire
        os.makedirs(dire,exist_ok=True)
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
        conditions = ['Neutral-OBSERVAR', 'Neg-OBSERVAR', 'Neg-REDUCIR', 'Neg-SUPRIMIR', 'Happy-OBSERVAR', 'Happy-SUPRIMIR', 'Happy-INCREMENTAR']
        sns.barplot(x=(conditions),y=attention_weight,color='pink')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.dire, 'condition_contributions_plot.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    #plot_pipe_plot will be made on canva
        

