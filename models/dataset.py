
import numpy as np;
from pathlib import Path;
import pandas as pd;
from torch_geometric.data import Data;
import torch;
from torch.utils.data import Dataset;
from config import config;
import sys;
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"));
from subject_filter import get_included_subjects;
class datasetPreparation(Dataset):
    def __init__(self, fm_only=False):
        super().__init__();
        self.datafolder = str(config.subjectDataFolder);
        self.datafolderPath = Path(self.datafolder);
        self.FCMatricesFilePath = "";
        self.TimeSeriesFilePath = "";
        self.clinicalCleanFilePath = str(config.clinicalCsv);
        self.subjectList = [];
        self.fm_only = bool(fm_only);
        self.conditionNames = config.conditions;
        self.clinicalDataFrame = None;
        self.clinicalLookup = {};
        try:
            self.clinicalDataFrame = pd.read_csv(self.clinicalCleanFilePath);
            self.clinicalDataFrame["subject_id"] = self.clinicalDataFrame["subject_id"].astype(str);
            self.clinicalLookup = self.clinicalDataFrame.set_index("subject_id").to_dict();
        except Exception as error:
            raise FileNotFoundError("clinical_clean.csv not found in the data folder. Please ensure it is present for proper dataset preparation.") from error;
        self.subjectData = [];
        self.rawNodeFeatures = None;
        self.nodeMean = None;
        self.nodeStd = None;
        self.DataList = self.execute();
    def organizeNodes(self):
        originalData = np.load(self.TimeSeriesFilePath);
        data = pd.DataFrame(originalData);
        dataset = [];
        for i in range(0, len(data.columns)):
            mean = data.iloc[:, i].mean();
            var = data.iloc[:, i].var(ddof=1);
            freq = self.calculateFrequencies(data.iloc[:, i]);
            dataset.append([mean, var, freq[0], freq[1], freq[2]]);
        dataset = pd.DataFrame(dataset);
        return torch.tensor(dataset.values, dtype=torch.float32);

    def calculateFrequencies(self, data):
        freq = np.fft.rfft(data, axis=0, norm="forward");
        freq = np.abs(freq) ** 2;
        labels = np.fft.rfftfreq(n=len(data), d=config.tr);
        indexes = [np.where((labels >= 0.01) & (labels < 0.04)), np.where((labels >= 0.04) & (labels < 0.1)), np.where((labels >= 0.1) & (labels < 0.25))];
        out = [np.sum(freq[index[0]]) for index in indexes];
        return out;

    def _resolvePath(self, subjectId, conditionName, kind):
        condName = self.cleanCondName(conditionName);
        if kind == "fc":
            candidates = [self.datafolderPath / subjectId / f"{subjectId}_FCMatrix_Cond_{condName}.npy", self.datafolderPath / subjectId / f"{subjectId}_FCMatrixCondition{condName}.npy", self.datafolderPath / subjectId / f"{subjectId}_FCMatrixCondition{conditionName.replace(' ', '')}.npy"];
        else:
            candidates = [self.datafolderPath / subjectId / f"{subjectId}_ROITimeSeries_Cond_{condName}.npy", self.datafolderPath / subjectId / f"{subjectId}_ROITimeSeries{condName}.npy", self.datafolderPath / subjectId / f"{subjectId}_ROITimeSeries{conditionName.replace(' ', '')}.npy"];
        for candidate in candidates:
            if candidate.exists():
                return str(candidate);
        raise FileNotFoundError(f"Could not find {kind} file for {subjectId} and condition {conditionName}");

    def edgeIndexAttr(self, FCMatrix=None):
        data = np.load(self.FCMatricesFilePath) if FCMatrix is None else FCMatrix;
        indexing = np.where(np.abs(data) >= np.percentile(np.abs(data), config.edgePercentile));
        indexList = np.array([indexing[0], indexing[1]]);  #where already sym
        edgeAttr = data[indexList[0], indexList[1]];
        packagedIndexAttr = [torch.tensor(indexList, dtype=torch.long), torch.tensor(edgeAttr, dtype=torch.float32)];
        return packagedIndexAttr;

    def getGroupLabel(self, subjectId):
        group = self.clinicalLookup.get("group", {}).get(subjectId, None);
        if group == "FM":
            return 0;
        elif group == "HC":
            return 1;
        else:
            return -1;
    def cleanCondName(self, cond):
        return cond.replace(" ", "").replace("-", "");

    def normalizeData(self, trainIndices=None):
        if self.rawNodeFeatures is None:
            self.rawNodeFeatures = [data.x.clone() for data in self.DataList];
        for data, raw in zip(self.DataList, self.rawNodeFeatures):
            data.x = raw.clone();
            data.x[:, 2:5] = torch.log1p(data.x[:, 2:5]);
        if trainIndices is None:
            trainIndices = range(len(self.subjectData));
        statGraphs = self.trainGraphsForStats(trainIndices);
        if len(statGraphs) == 0:
            return;
        stacked = torch.stack([data.x for data in statGraphs]);
        nodeMean = stacked.mean(dim=0);
        nodeStd = stacked.std(dim=0);
        self.nodeMean = nodeMean;
        self.nodeStd = nodeStd;
        for data in self.DataList:
            data.x = (data.x - nodeMean)/(nodeStd + 1e-8);

    def applyNormalization(self, nodeMean, nodeStd):
        if self.rawNodeFeatures is None:
            self.rawNodeFeatures = [data.x.clone() for data in self.DataList];
        for data, raw in zip(self.DataList, self.rawNodeFeatures):
            data.x = raw.clone();
            data.x[:, 2:5] = torch.log1p(data.x[:, 2:5]);
            data.x = (data.x - nodeMean)/(nodeStd + 1e-8);
        self.nodeMean = nodeMean;
        self.nodeStd = nodeStd;

    def trainGraphsForStats(self, trainIndices):
        graphs = [];
        for index in trainIndices:
            graphs.extend(self.subjectData[index]["graphs"]);
        return graphs;

    def execute(self):
        dataList = [];
        subjectData = [];
        clinical = pd.read_csv(self.clinicalCleanFilePath);
        clinical["group"] = clinical["group"].map({"FM": 0, "HC": 1});  #pandas3: map instead of int-into-str-dtype assignment
        subjectGraphs = {};
        inc = set(get_included_subjects());  
        for subfolder in self.datafolderPath.iterdir():
            if subfolder.is_dir() and subfolder.name in inc:
                if self.fm_only and self.getGroupLabel(subfolder.name) != 0:
                    continue;
                self.subjectList.append(subfolder.name);
                subjectGraphs[subfolder.name] = [];
                for i in range(0, len(self.conditionNames)):
                    self.FCMatricesFilePath = self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_FCMatrixCondition" + self.conditionNames[i].replace(" ", "") + ".npy";
                    self.TimeSeriesFilePath = self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_ROITimeSeries" + self.conditionNames[i].replace(" ", "") + ".npy";
                    unpackage = self.edgeIndexAttr();
                    yVal = clinical.loc[clinical["subject_id"] == subfolder.name, "group"].iloc[0];
                    data = Data(x=self.organizeNodes(), y=torch.tensor(yVal, dtype=torch.long), edge_index=unpackage[0], edge_attr=unpackage[1]);
                    data.subjectID = subfolder.name;
                    data.condition = self.conditionNames[i];
                    subjectGraphs[subfolder.name].append(data);
                    assert data.x.shape[0] == config.nNodes, f"Subject {subfolder.name} condition {self.conditionNames[i]} has {data.x.shape[0]} nodes, expected {config.nNodes}";
                    dataList.append(data);
        for subjectId, graphs in subjectGraphs.items():
            subjectData.append({"subject_id": subjectId, "graphs": graphs, "group_label": torch.tensor([self.getGroupLabel(subjectId)], dtype=torch.long)});
        self.DataList = dataList;
        self.subjectData = subjectData;
        self.normalizeData();
        return dataList;

    def __len__(self):
        return len(self.DataList);

    def __getitem__(self, index):
        return self.DataList[index];