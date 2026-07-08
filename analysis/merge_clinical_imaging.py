import pandas as pd
import numpy as np
from config import config 

def merge_clinical_imaging(csv,embed):
    df=pd.read_csv(csv)
    embed={k: v.detach().cpu().numpy() if hasattr(v, 'detach') else v for k, v in embed.items()}
    df2=pd.DataFrame.from_dict(embed,orient='index')
    df2.columns=[f'emb_{i}' for i in range(config.D_MODEL)] 
    df2=df2.reset_index().rename(columns={'index': 'subject_id'})
    merged=pd.merge(df, df2, on='subject_id')
    return merged