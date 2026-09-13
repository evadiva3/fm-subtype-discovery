from config import config;
from pathlib import Path;
import pandas as pd;
import numpy as np;
import driver_utils as driverUtils;
import warnings;
class Orchestrator():
    def __init__(self, subjectExclusions = None, embeddingsPath = None, labelPath = None, clinicalCSV = None, dumpPath = None):
        if embeddingsPath is not None:
            try:
                self.embeddings = np.load(embeddingsPath);
            except FileNotFoundError:
                warnings.warn(f"Path Specified: {embeddingsPath} Does Not Exist - Check File Type. Using Default Path");
                self.embeddings = np.load(config.embeddingPath);
        else:
            self.embeddings = np.load(config.embeddingPath);
        if labelPath is not None:
            try:
                self.labels = pd.read_csv(labelPath).set_index("Subject_Id", drop=False);
            except FileNotFoundError:
                warnings.warn(f"Path Specified: {labelPath} Does Not Exist - Check File Type. Using Default Path");
                self.labels = pd.read_csv(config.kLabelPath).set_index("Subject_Id", drop=False);
        else:
            self.labels = pd.read_csv(config.kLabelPath).set_index("Subject_Id", drop=False);
        if subjectExclusions is not None:
            try:
                self.subjectExclusions = pd.read_csv(subjectExclusions).set_index("subject_id", drop=False);
            except FileNotFoundError:
                warnings.warn(f"Path Specified: {subjectExclusions} Does Not Exist - Check File Type. Using Default Path");
                self.subjectExclusions = pd.read_csv(config.exclusionManifestPath).set_index("subject_id", drop=False);
        else:
            self.subjectExclusions = pd.read_csv(config.exclusionManifestPath).set_index("subject_id", drop=False);
        self.ids = self.labels["Subject_Id"];
        if clinicalCSV is not None:
            try:
                self.clinicalCSV = pd.read_csv(clinicalCSV).set_index("subject_id", drop=False);
            except FileNotFoundError:
                warnings.warn(f"Path Specified: {clinicalCSV} Does Not Exist - Check File Type. Using Default Path");
                self.clinicalCSV = pd.read_csv(config.clinicalCsv).set_index("subject_id", drop=False);
        else:
            self.clinicalCSV = pd.read_csv(config.clinicalCsv).set_index("subject_id", drop=False);
        self.savePath = Path(dumpPath) if dumpPath is not None else config.analysisOrchestrator;
        self.savePath.parent.mkdir(parents=True, exist_ok=True);
    def effectiveRank(self):
        effectiveRank, pc1 = driverUtils.eff_rank(self.embeddings);
        return [effectiveRank, pc1];
    def leaveOneOut(self):
        y = driverUtils.severity_y(self.ids);
        ridge = driverUtils._ridge;
        r, r2 = driverUtils.score_r(driverUtils.loo_predict(ridge, self.embeddings, y),y);
        permutations = driverUtils.perm_r(ridge, self.embeddings, y, r);
        return[r, r2, permutations];
    def kToFD(self):
        from scipy import stats;
        subjectExclusions = self.subjectExclusions.loc[(self.subjectExclusions["excluded"]==False),:];
        fmIds = self.clinicalCSV.index[(self.clinicalCSV["group"]=="FM")];
        subjectExclusions = subjectExclusions.loc[subjectExclusions.index.intersection(fmIds), :];
        fdColumns = list(subjectExclusions.columns[config.cMeanFDStartIdx:config.cMeanFDEndIdx]);
        mean = subjectExclusions[fdColumns].apply(pd.to_numeric, errors="coerce").mean(axis=1);
        labelSeries = self.labels.set_index("Subject_Id")["Label"];
        subjectFD = pd.DataFrame({"SubjectId":subjectExclusions.index, "FD": mean.to_numpy(), "K":labelSeries.reindex(subjectExclusions.index).to_numpy()});
        subjectFD = subjectFD.dropna(subset=["FD","K"]).reset_index(drop=True);
        clusters = [group["FD"].to_numpy() for _, group in subjectFD.groupby("K")];
        if len(clusters)<2:
            return [float("nan"), float("nan"), len(subjectFD["SubjectId"])];
        H, kruskalP = stats.kruskal(*clusters);
        return [H, kruskalP, len(subjectFD["SubjectId"])];
    def main(self):
        package = self.effectiveRank();
        package1 = self.leaveOneOut();
        package2 = self.kToFD();
        pd.DataFrame({"effectiveRank": package[0], "pc1": package[1], "severityR": package1[0], "severityR2": package1[1], "severityPermutations": package1[2], "hStat": package2[0], "kruskalP": package2[1], "nCount": package2[2]}, index=[0]).to_csv(self.savePath);
if __name__ == "__main__":
    orchestrate = Orchestrator(None, None, None, None, None);
    orchestrate.main();
