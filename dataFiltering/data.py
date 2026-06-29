import pandas as pd;
pathToXML ="../data/Clinical_fm_66.xlsx"
pathToCSV = "../data/Cinical_fm_66_CSV.csv"
def getStatData():
    data = pd.read_excel(pathToXML, sheet_name="data_66", engine ="openpyxl");
    data = data.drop(data[data['rid'].isin([5, 12, 32])].index);

    columns = {
        'rid': 'subject_id',
        'gp': 'group',
        '1_age': 'age',
        '1_vas_pain_iv': 'vas_pain',
        '6_hamd_total': 'hamd_total',
        '6_hamd_cat': 'hamd_cat',
        '7_hama_total': 'hama_total',
        '11_ts': 'tas_total',
        '11_ id': 'tas_dif',
        '11_df': 'tas_ddf',
        '11_eot': 'tas_eot',
        '11_cat': 'alexithymia',
        '10_R': 'erq_reappraisal',
        '10_S': 'erq_suppression',
    }
    filteredList = data[['rid','gp','1_age','1_vas_pain_iv','6_hamd_total','6_hamd_cat','7_hama_total','11_ts','11_ id', '11_df','11_eot', '11_cat','10_R','10_S']].copy();
    filteredList = filteredList.rename(columns=columns);
    filteredList['subject_id'] =  "sub-"+filteredList['subject_id'].astype(str).str.zfill(3);
    filteredList.loc[filteredList['group']== 0,'group'] = "FM";
    filteredList.loc[filteredList['group']==1,'group'] = "HC";
    data.to_csv(pathToCSV, index=False);