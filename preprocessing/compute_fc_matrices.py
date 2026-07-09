# changes made:
# removed orphaned buildSparse graph-construction path

import os;
import numpy as np;
import pandas as pd;
from pathlib import Path;
from nilearn.maskers import NiftiLabelsMasker;
from nilearn.connectome import ConnectivityMeasure;
from sklearn.covariance import LedoitWolf;
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

    def execute(self):
        try:
            for subfolder in self.dataFolderPath.iterdir():
                if subfolder.is_dir() and not subfolder.name.startswith("top_") and subfolder.name != "excluded":
                    self.pathToBOLDFile = os.path.join(self.dataFolder, subfolder.name, f"{subfolder.name}_BOLD.nii.gz");
                    self.pathToConfoundsFile = os.path.join(self.dataFolder, subfolder.name, f"{subfolder.name}_Confounds.tsv");
                    self.eventsListPath = os.path.join(self.dataFolder, subfolder.name, f"{subfolder.name}_events.tsv");
                    fcMatrices = self.buildFCMatrices();
                    outputDir = os.path.join(self.dataFolder, subfolder.name);
                    os.makedirs(outputDir, exist_ok=True);
                    for i in range(len(fcMatrices)):
                        condName = self.conditions[i].replace(" ", "");
                        np.save(os.path.join(outputDir, f"{subfolder.name}_FCMatrixCondition{condName}.npy"), fcMatrices[i]);
                        if self.saveTimeSeries is not None:
                            np.save(os.path.join(outputDir, f"{subfolder.name}_ROITimeSeries{condName}.npy"), self.saveTimeSeries[i]);
        except Exception as error:
            raise RuntimeError("Check Pathname/Pipeline Parameters") from error;