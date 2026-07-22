from config import config;
import pandas as pd;
import numpy as np;
import driver_utils as driverUtils;
import warnings;
class Orchestrator():
    def __init__(self, subjectExclusions = None, embeddingsPath = None, labelPath = None, clinicalCSV = None):
        if embeddingsPath is not None:
            try:
                self.embeddings = pd.DataFrame(np.load(embeddingsPath));
            except FileNotFoundError:
                warnings.warn(f"Path Specified: {embeddingsPath} Does Not Exist - Check File Type. Using Default Path");
                self.embeddings = pd.DataFrame(np.load(config.embeddingPath));
        else:
            self.embeddings = pd.DataFrame(np.load(config.embeddingPath));
        if labelPath is not None:
            try:
                self.labels = pd.read_csv(labelPath);
            except FileNotFoundError:
                warnings.warn(f"Path Specified: {labelPath} Does Not Exist - Check File Type. Using Default Path");
                self.labels = pd.read_csv(config.kLabelPath);
        else:
            self.labels = pd.read_csv(config.kLabelPath);
        if subjectExclusions is not None:
            try:
                self.subjectExclusions = pd.read_csv(subjectExclusions)
            except FileNotFoundError:
                warnings.warn(f"Path Specified: {subjectExclusions} Does Not Exist - Check File Type. Using Default Path");
                self.subjectExclusions = pd.read_csv(config.exclusionManifestPath);
        else:
            self.subjectExclusions = pd.read_csv(config.exclusionManifestPath);
        self.ids = self.labels["Subject_Id"];
        if clinicalCSV is not None:
            try:
                self.clinicalCSV = pd.read_csv(clinicalCSV);
            except FileNotFoundError:
                warnings.warn(f"Path Specified: {clinicalCSV} Does Not Exist - Check File Type. Using Default Path");
                self.clinicalCSV = pd.read_csv(config.clinicalCSV);
        else:
            self.clinicalCSV = pd.read_csv(config.clinicalCSV);
    def effectiveRank(self):
        effectiveRank, pc1 = driverUtils.eff_rank(self.embeddings);
        data = {"effectiveRank": effectiveRank, "pc1": pc1};
        return pd.DataFrame(data);
    def leaveOneOut(self):
        y = driverUtils.severity_y(self.ids);
        ridge = driverUtils._ridge;
        r, r2 = driverUtils.score_r(driverUtils.loo_predict(ridge, self.embeddings, y));
        permutations = driverUtils.perm_r(ridge, self.embeddings, y, r);
        return[r, r2, permutations];
    def kToFD(self):
        from scipy import stats;
        mean = [];
        subjectExclusions = self.subjectExclusions.loc[(self.subjectExclusions["excluded"]==False),:];
        subjectExclusions["k"] = self.labels.loc[self.labels["Subject_Id"], "Label"];
        for subject in subjectExclusions.to_numpy():
            mean.append(subject[config.cMeanFDStartIdx:config.cMeanFDEndIdx].mean());
        subjectFD = pd.DataFrame({"SubjectId":subjectExclusions["subject_id"], "FD": mean, "K":subjectExclusions["k"]});
        subjectFD = subjectFD.loc[(self.clinicalCSV["group"]=="FC"), :];
        for _ in range(0,len(subjectFD["SubjectId"])):
            for i in range(0,len(subjectFD["SubjectId"])-1):
                if (subjectFD.iloc[i,1]>subjectFD.iloc[i+1,1]):
                    subjectFD.iloc[i,:], subjectFD.iloc[i+1,:] = subjectFD.iloc[i+1,:].copy(), subjectFD.iloc[i,:].copy();
        subjectFD["rankings"] = subjectFD.index+1;
        clusters = [subjectFD[subjectFD["K"]==k] for k in config.kmeansKRange];
        H = ((12/((len(subjectFD["SubjectId"]))*(len(subjectFD["SubjectId"])+1)))*(sum(((sum(group["rankings"])**2)/len(group["SubjectId"]) for group in clusters)))) - 3*(len(subjectFD["SubjectId"])+1);
        HPerm = stats.chi2.sf(H,len(clusters)-1);
        return [H, HPerm, len(subjectFD["SubjectId"])];
    def main(self):
        package = self.effectiveRank();
        package1 = self.leaveOneOut();
        package2 = self.kToFD();
        pd.DataFrame({"effectiveRank": package, "severityR": package1[0], "severityR2": package1[1], "severityPermutations": package1[2], "hStat": package2[0], "hPerm": package2[1], "nCount": package2[2]}).to_csv(config.analysisOrchestrator);