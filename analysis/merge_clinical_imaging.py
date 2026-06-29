import pandas as pd
import numpy as np

def merge_clinical_imaging(csv,embed):
    df=pd.read_csv(csv)
    df2=pd.DataFrame.from_dict(embed,orient='index')
    df2.columns = [f'emb_{i}' for i in range(64)]
    df2 = df2.reset_index().rename(columns={'index': 'subject_id'})
    merged = pd.merge(df, df2, on='subject_id')
    return merged