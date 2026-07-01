import numpy as np
import pandas as pd
import torch

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


    def subtype_importance(self,attention_w,subtype_label):

