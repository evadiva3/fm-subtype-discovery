import numpy as np;
from pathlib import Path;
import pandas as pd;
from torch_geometric.data import Data
import torch;
from torch.utils.data import Dataset
class datasetPreparation(Dataset):
  #still needs to load clinical_clean.csv for the HC/FM status for data.y to keep track of who actually has ts and who doesn't. Other than that, should be done.  
  def __init__(self):
    super().__init__();
    self.datafolder = "../pathname";
    self.datafolderPath = Path(self.datafolder);
    self.FCMatricesFilePath = "";
    self.TimeSeriesFilePath = "";
    self.conditionNames = ["Neutral - OBSERVAR", "Negativo - OBSERVAR", "Happy - OBSERVAR", "Negativo - REDUCIR", "Negativo - SUPRIMIR", "Happy - SUPRIMIR", "Happy - INCREMENTAR"]
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
  def execute(self):
    dataList = [];
    for subfolder in self.datafolderPath.iterdir():
      if subfolder.is_dir():
        for i in range(0,7):
          self.FCMatricesFilePath = self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_FCMatrixCondition" + self.conditionNames[i].replace(" ", "") + ".npy";
          self.TimeSeriesFilePath = self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_ROITimeSeries" + self.conditionNames[i].replace(" ", "") + ".npy";
          unpackage = self.edgeIndexAttr();
          data = Data(x=self.organizeNodes(), edge_index = unpackage[0], edge_attr = unpackage[1]);
          data.subjectID = subfolder.name;
          data.condition = self.conditionNames[i];
          dataList.append(data);
    return dataList;
  def __len__(self):
    return len(self.DataList);
  def __getitem__(self, index):
    return self.DataList[index];