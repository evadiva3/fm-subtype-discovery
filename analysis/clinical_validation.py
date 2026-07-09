import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
from config import config  


class clinical_validator:
    def __init__(self,clinical_csv_path=None):
        clinical_csv_path=config.clinicalCsv if clinical_csv_path is None else clinical_csv_path
        self.df=pd.read_csv(clinical_csv_path)
        self.count_vars=['age', 'vas_pain', 'hamd_total', 'hama_total', 'tas_total', 
        'tas_dif', 'tas_ddf', 'tas_eot', 'erq_reappraisal', 'erq_suppression']


    def compare_groups(self, subtype):
        self.df['temp']=subtype
        group=self.df[self.df['temp']==0]
        group1=self.df[self.df['temp']==1]
        results=[]
        for i in self.count_vars:
            stat, p=stats.mannwhitneyu(group[i], group1[i], alternative='two-sided')
            #above is man-whitney u math
            mean0=group[i].mean()
            mean1=group1[i].mean()
            std0=group[i].std()
            std1=group1[i].std()
            pooled_std = np.sqrt((std0**2 + std1**2) / 2)
            d=(mean0 - mean1)/pooled_std
            results.append({
                'variable': i,
                'p_value': p,
                'cohens_d': d
                            })
            #above is cohen
        results_df=pd.DataFrame(results)
        _,corrected_p, _, _=multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_corrected']=corrected_p
        results_df['significant']=corrected_p < config.fdrAlpha
        return results_df
    
    def compute_effect_sizes(self,subtype):
        self.df['temp']=subtype
        group=self.df[self.df['temp']==0]
        group1=self.df[self.df['temp']==1]
        effect_sizes={}
        for i in self.count_vars:
            mean=group[i].mean()
            mean1=group1[i].mean()
            std=group[i].std()
            std1=group1[i].std()
            pooled_std=np.sqrt((std**2 + std1**2) / 2)
            effect_sizes[i]=(mean - mean1)/pooled_std
        return effect_sizes



    def run_all(self,subtype,save_dir=None):
        save_dir=config.resultsRoot if save_dir is None else save_dir
        os.makedirs(save_dir, exist_ok=True)
        df=self.compare_groups(subtype)
        effect_sizes=self.compute_effect_sizes(subtype)
        df.to_csv(os.path.join(save_dir, 'clinical_validation_results.csv'), index=False)
        print(effect_sizes)
        return df
    

        





