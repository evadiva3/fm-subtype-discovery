
import numpy as np;
from pathlib import Path;
import pandas as pd;
from torch_geometric.data import Data
import torch;
from torch.utils.data import Dataset
class datasetPreparation(Dataset):
  #still needs to load clinical_clean.csv for the HC/FM status for data.y to keep track of who actually has ts and who doesn't. Other than that, should be done.  
  def __init__(self, fm_only=False):
    super().__init__();
    self.datafolder = "../pathname";
    self.datafolderPath = Path(self.datafolder);
    self.FCMatricesFilePath = "";
    self.TimeSeriesFilePath = "";
    self.subjectList = [];
    self.fm_only=fm_only
    self.conditionNames = ["Neutral - OBSERVAR", "Negativo - OBSERVAR", "Happy - OBSERVAR", "Negativo - REDUCIR", "Negativo - SUPRIMIR", "Happy - SUPRIMIR", "Happy - INCREMENTAR"]
    #load clincal tracking for HC/FM diagnosis
    self.clinical_csv_path=self.datafolderPath / "clinical_clean.csv"
    if self.clinical_csv_path.exists():
      self.c_df=pd.read_csv(self.clinical_csv_path)
      self.c_df['subject_id']=self.c_df['subject_id'].astype(str)
      self.c_look=self.c_df.set_index('subject_id').to_dict()
    else:
      print("clinical_clean.csv not found in the data folder. Please ensure it is present for proper dataset preparation."  )
      self.c_look={}
    self.DataList = self.execute();
  
  def organizeNodes(self):
    originalData = np.load(self.TimeSeriesFilePath);
    data = pd.DataFrame(originalData);
    dataset = [];
    for i in range (0,len(data.columns)):
      mean = data.iloc[:,i].mean();
      var = data.iloc[:,i].var(ddof=1);
      freq = self.calculateFrequencies(data.iloc[:,i]);
      dataset.append([mean, var, freq[0], freq[1], freq[2]]);
    dataset = pd.DataFrame(dataset);
    return torch.tensor(dataset.values,dtype = torch.float32);

  def calculateFrequencies(self, data):
    freq = np.fft.rfft(data, axis = 0, norm="forward");
    freq = np.abs(freq)**2;
    labels = np.fft.rfftfreq(n=len(data), d = 2);
    indexes = [np.where((labels>=0.01) & (labels<0.04)), np.where((labels>=0.04) & (labels<0.1)), np.where((labels>=0.1) & (labels<0.25))];
    out = [np.sum(freq[index[0]]) for index in indexes];
    return out;

  def edgeIndexAttr(self):
    data = np.load(self.FCMatricesFilePath);
    indexing = np.where(np.abs(data)>=np.percentile(np.abs(data), 80));
    indexList = np.array([indexing[0],indexing[1]]);
    indexList = np.concatenate((indexList, indexList[::-1]), axis = 1);
    edgeAttr = data[indexList[0], indexList[1]];
    packagedIndexAttr = [torch.tensor(indexList, dtype=torch.long), torch.tensor(edgeAttr, dtype= torch.float32)];
    return packagedIndexAttr;

#helps resolve FM=0 HC=1 from clinical lookup if not there = -1
  def get_group_label(self, subject_id):
    group=self.c_look.get('group', {}).get(subject_id, None)
    if group=='FM':
      return 0
    elif group=='HC':
      return 1
    else:
      return -1  #not there/unknown
  
  #strip space and dash from condition name
  def clean_cond_name(self,cond):
    return cond.replace(" ", "").replace("-", "")
  
  def execute(self):
    dataList = [];
    for sub in self.datafolderPath.iterdir():
      if not sub.is_dir():
        continue
      subject_id=sub.name
      group_label=self.get_group_label(subject_id)
      if self.fm_only and group_label!=0:
        continue
      sub_graphs=[]
      valid=True
      for i in range(0,7):
        cond_name=self.clean_cond_name(self.conditionNames[i])
        self.FCMatricesFilePath=str(
          self.datafolderPath / sub.name /
          f"{sub.name}_FCMatrix_Cond_{cond_name}.npy"
        )
        self.TimeSeriesFilePath=str(
          self.datafolderPath / sub.name /
          f"{sub.name}_ROITimeSeries_Cond_{cond_name}.npy"
        )
        if not Path(self.FCMatricesFilePath).exists() or \
          not Path(self.TimeSeriesFilePath).exists():
            print(f"Missing files: {subject_id} condition {cond_name} skipping subject")
            valid=False
            break
        unpackage=self.edgeIndexAttr()
        graph_data=Data(
          x=self.organizeNodes(),
          edge_index=unpackage[0],
          edge_attr=unpackage[1]
        )
        graph_data.subject_id=subject_id
        graph_data.condition=self.conditionNames[i]
        graph_data.y=torch.tensor([group_label], dtype=torch.long)
        sub_graphs.append(graph_data)

      if valid and len(sub_graphs)==7:
        dataList.append({
          'subject_id': subject_id,
          'graphs': sub_graphs,
          'group_label': torch.tensor([group_label], dtype=torch.long)
        })
        self.subjectList.append(subject_id)

    return dataList

  def __len__(self):
    return len(self.DataList);

  def __getitem__(self, index):
    return self.DataList[index];