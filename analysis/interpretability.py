import numpy as np
import pandas as pd
import torch
from nilearn import datasets

class attention_interpreter():
    def __init__ (self,model,attention_pool,dataloader,device):
        self.model=model
        self.attention_pool=attention_pool
        self.dataloader=dataloader
        self.device=device
        model.eval()
        attention_pool.eval()
    def extract_node_attention(self):
        self.model.eval()
        results={}
        with torch.no_grad():
            for i in self.dataloader:
                i = i.to(self.device) 
                output=self.model(i)
                for sub in range(len(i.subject_id)):              
                    a=i.subject_id[sub]
                    results[a]=output[sub]
                
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
        a=datasets.fetch_atlas_schaefer_2018(n_rois=200)
        label=a['labels']
        networks={}
        for idx, label_str in enumerate(label):
            network=label_str.split('_')[2]
            if network not in networks:
                networks[network] = []
            networks[network].append(importance[idx])
        for i in networks:
            networks[i]=np.mean(networks[i],axis=0)
        return networks
    
    def get_condition_weights(self):
        self.model.eval()
        aw=[]
        with torch.no_grad():
            for i in self.dataloader:
                i=i.to(self.device)
                embedd=self.model(i)
                pooled, weights=self.attention_pool(embedd)
                aw.append(weights)
        return torch.stack(aw).mean(dim=0)

        

