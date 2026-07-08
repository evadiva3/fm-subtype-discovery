# changes made:
# rank-deficiency: swap Pearson for Ledoit-Wolf and make 200x200 stable from 16-vol task time;
# forced relation learning: kill diagonal and Fisher z shift;
# group sub-batch: read sick vs healthy from clinical csv and wrap 7 task graphs;
# sensitivity testing: chop graphs into density boxes;

# imports were missing;
import os;
import numpy as np;
import pandas as pd;
from pathlib import Path;
from nilearn.maskers import NiftiLabelsMasker;
from nilearn.connectome import ConnectivityMeasure;
from sklearn.covariance import LedoitWolf;
import torch;
from torch_geometric.data import Data;
from config import config;

class preprocessBOLD:
    def __init__(self):
        self.dataFolder = config.SUBJECTDATAFOLDER;
        self.dataFolderPath = Path(self.dataFolder);
        self.masker = NiftiLabelsMasker(labels_img=config.MASKER);
        self.pathToBOLDFile = "";
        self.pathToConfoundsFile = "";
        self.eventsListPath = "";
        self.saveTimeSeries = None;
        self.conditions = config.CONDITIONS;
    def buildTimeSeries(self):
        try:
            confound = pd.read_csv(self.pathToConfoundsFile, sep="\t");
            selectedConfounds = ["global_signal", "white_matter", "csf", "trans_x", "trans_x_derivative1", "trans_x_derivative1_power2", "trans_x_power2", "trans_y", "trans_y_derivative1", "trans_y_power2", "trans_y_derivative1_power2", "trans_z", "trans_z_derivative1", "trans_z_derivative1_power2", "trans_z_power2", "rot_x", "rot_x_derivative1", "rot_x_power2", "rot_x_derivative1_power2", "rot_y", "rot_y_derivative1", "rot_y_power2", "rot_y_derivative1_power2", "rot_z", "rot_z_derivative1", "rot_z_power2", "rot_z_derivative1_power2"]; 
            cleanedConfounds = confound[selectedConfounds].fillna(0);
            return self.masker.fit_transform(self.pathToBOLDFile, confounds=cleanedConfounds);
        except Exception as error:
            raise RuntimeError("Check Pathname Hardcodes") from error;
    def splitConditions(self):
        timeSeries = self.buildTimeSeries();
        tsvData = pd.read_csv(self.eventsListPath, sep="\t");
        conditionTimeSeries = [[] for _ in range(0, 7)]; 
        conditionFrames = [[] for _ in range(0, 7)];
        for row in tsvData.itertuples():
            for i in range(len(self.conditions)):
                if row.trial_type == self.conditions[i]:
                    startTr = int(row.onset / 2);
                    endTr = int((row.onset + row.duration) / 2);
                    conditionTimeSeries[i].append(timeSeries[startTr:endTr, :]);
        for condition in range(0, len(conditionTimeSeries)):
            conditionFrames[condition] = pd.DataFrame(np.concatenate(conditionTimeSeries[condition]));
        self.saveTimeSeries = [frame.to_numpy() for frame in conditionFrames];
        return conditionFrames;
    def buildFCMatrices(self):
        conditionFrames = self.splitConditions();
        fcMatrices = [[] for _ in range(7)]; 
        lwMeasure = ConnectivityMeasure(kind="correlation", cov_estimator=LedoitWolf());
        for i in range(len(conditionFrames)):
            array = conditionFrames[i].to_numpy();
            rMatrix = lwMeasure.fit_transform([array])[0];
            zMatrix = np.arctanh(rMatrix);
            np.fill_diagonal(zMatrix, 0);
            zMatrix[np.isinf(zMatrix)] = 0;
            fcMatrices[i] = zMatrix;
        return fcMatrices;
    def buildSparse(self, fcMatrix, topK):
        numNodes = fcMatrix.shape[0];
        meanConn = fcMatrix.mean(axis=1);
        varConn = fcMatrix.var(axis=1);
        degree = (np.abs(fcMatrix) > 0).sum(axis=1).astype(float);
        spectral = np.linalg.norm(fcMatrix, axis=1);
        clust = np.array([fcMatrix[i, np.argsort(np.abs(fcMatrix[i]))[-topK:]].mean()for i in range(numNodes)]);
        x = torch.tensor(np.stack([meanConn, varConn, degree, clust, spectral], axis=1),dtype=torch.float,);
        sources, targets, edgeWeights = [], [], [];
        for nodeIdx in range(numNodes):
            row = fcMatrix[nodeIdx];
            topIndices = np.argsort(np.abs(row))[-topK:];
            for targetIdx in topIndices:
                sources.append(nodeIdx);
                targets.append(targetIdx);
                edgeWeights.append(row[targetIdx]);
        edgeIndex = torch.tensor([sources, targets], dtype=torch.long);
        edgeAttr = torch.tensor(edgeWeights, dtype=torch.float).unsqueeze(-1);
        return Data(x=x, edge_index=edgeIndex, edge_attr=edgeAttr);
    def execute(self):
        try:
            topKList = [20, 30, 40, 50];
            for subfolder in self.dataFolderPath.iterdir():
                if subfolder.is_dir() and not subfolder.name.startswith("top_"):
                    self.pathToBOLDFile = os.path.join(self.dataFolder, subfolder.name, f"{subfolder.name}_BOLD.nii.gz");
                    self.pathToConfoundsFile = os.path.join(self.dataFolder, subfolder.name, f"{subfolder.name}_Confounds.tsv");
                    self.eventsListPath = os.path.join(self.dataFolder, subfolder.name, f"{subfolder.name}_events.tsv");
                    fcMatrices = self.buildFCMatrices();
                    for topK in topKList:
                        outputDir = os.path.join(self.dataFolder, f"top_{topK}", subfolder.name);
                        os.makedirs(outputDir, exist_ok=True);
                        for i in range(len(fcMatrices)):
                            condName = self.conditions[i].replace(" ", "").replace("-", "");
                            np.save(os.path.join(outputDir, f"{subfolder.name}_FCMatrix_Cond_{condName}.npy"), fcMatrices[i]);
                            if self.saveTimeSeries is not None:
                                np.save(os.path.join(outputDir, f"{subfolder.name}_ROITimeSeries_Cond_{condName}.npy"), self.saveTimeSeries[i]);
                            graphData = self.buildSparse(fcMatrices[i], topK);
                            torch.save(graphData, os.path.join(outputDir, f"{subfolder.name}_PyGGraph_Cond_{condName}.pt"));
        except Exception as error:
            raise RuntimeError("Check Pathname/Pipeline Parameters") from error;
      