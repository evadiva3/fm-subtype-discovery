import numpy as np;
from pathlib import Path;
import pandas as pd;
from torch_geometric.data import Data
import torch;
from torch.utils.data import Dataset
class datasetPreparation(Dataset):
  def __init__(self, avgCond : bool):
    super().__init__();
    self.datafolder = "../pathname";
    self.datafolderPath = Path(self.datafolder);
    self.FCMatricesFilePath = "";
    self.TimeSeriesFilePath = "";
    self.clinicalCleanFilePath = "";
    self.subjectList = [];
    self.conditionNames = ["Neutral - OBSERVAR", "Negativo - OBSERVAR", "Happy - OBSERVAR", "Negativo - REDUCIR", "Negativo - SUPRIMIR", "Happy - SUPRIMIR", "Happy - INCREMENTAR"]
    self.avgCond = avgCond;
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
  def edgeIndexAttr(self, loadFile, FCMatrix):
    if loadFile:
      data = np.load(self.FCMatricesFilePath);
    else:
      data = FCMatrix;
    indexing = np.where(np.abs(data)>=np.percentile(np.abs(data), 80));
    indexList = np.array([indexing[0],indexing[1]]);
    indexList = np.concatenate((indexList, indexList[::-1]), axis = 1);
    edgeAttr = data[indexList[0], indexList[1]];
    packagedIndexAttr = [torch.tensor(indexList, dtype=torch.long), torch.tensor(edgeAttr, dtype= torch.float32)];
    return packagedIndexAttr;
  def averageConditions(self, subfolder, clinical):
    if subfolder.is_dir() and not subfolder.name.startswith("top_"):
      fullFC = [];
      nodes = [];
      yVal = clinical.loc[clinical['subject_id']==subfolder.name, 'group'].iloc[0];
      for g in range(0,7):
        fullFC.append(np.load(self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_FCMatrixCondition" + self.conditionNames[g].replace(" ", "") + ".npy"))
        self.TimeSeriesFilePath = self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_ROITimeSeries" + self.conditionNames[g].replace(" ", "") + ".npy";
        nodes.append(self.organizeNodes().numpy());
      fullFC = np.stack(fullFC, axis=0);
      fullFC = np.mean(fullFC, axis=0);
      nodes = np.stack(nodes, axis=0);
      nodes = np.mean(nodes, axis=0);
      packaged = self.edgeIndexAttr(False, fullFC);
      graphData = Data(x=torch.tensor(nodes, dtype=torch.float32), y = torch.tensor(yVal, dtype=torch.long), edge_attr=packaged[1], edge_index=packaged[0]);
      graphData.subjectID = subfolder.name;
    return graphData;
  def execute(self):
    dataList = [];
    clinical = pd.read_csv(self.clinicalCleanFilePath);
    clinical.loc[clinical['group']== "FM",'group'] = 0;
    clinical.loc[clinical['group']=="HC",'group'] = 1;
    if(self.avgCond ==False):
      for subfolder in self.datafolderPath.iterdir():
        if subfolder.is_dir():
          self.subjectList.append(subfolder.name);
          for i in range(0,7):
            self.FCMatricesFilePath = self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_FCMatrixCondition" + self.conditionNames[i].replace(" ", "") + ".npy";
            self.TimeSeriesFilePath = self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_ROITimeSeries" + self.conditionNames[i].replace(" ", "") + ".npy";
            unpackage = self.edgeIndexAttr(True, None);
            yVal = clinical.loc[clinical['subject_id']==subfolder.name, 'group'].iloc[0];
            data = Data(x=self.organizeNodes(), y = torch.tensor(yVal, dtype=torch.long), edge_index = unpackage[0], edge_attr = unpackage[1]);
            data.subjectID = subfolder.name;
            data.condition = self.conditionNames[i];
            dataList.append(data);
    else:
      for subfolder in self.datafolderPath.iterdir():
        if subfolder.is_dir():
          self.subjectList.append(subfolder.name);
          dataList.append(self.averageConditions(subfolder, clinical)); 
    return dataList;
  def __len__(self):
    return len(self.DataList);
  def __getitem__(self, index):
    return self.DataList[index];