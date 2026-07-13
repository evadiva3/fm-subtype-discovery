#changes made: 
#wiring evaluate py
#gap stat fix 
#centriod distance
#no hardcode


import os;
import json;
import torch;
import pandas as pd;
import numpy as np;
from sklearn.cluster import KMeans;
from sklearn.metrics import silhouette_score;
import umap;
from pathlib import Path;
import sys;

_ROOT=Path(__file__).resolve().parent.parent;
for _p in (_ROOT, _ROOT / "src", _ROOT / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p));

from torch.utils.data import Dataset;
from torch_geometric.data import Batch;
from dataset import datasetPreparation;
from analysis.evaluate import cluster_evaluate;
from config import config;

class cluster():
    def __init__(self, GNNEncoder, directory, conditionList, subjectList):
        self.GNNEncoder = GNNEncoder;
        self.PTPath = directory;
        self.subjectDList = [[None for _ in range(0, len(conditionList))] for _ in range(0, len(subjectList))];
        self.conditionList = conditionList;
        self.subjectList = subjectList;

    def deploy(self, dataloader):
        self.subjectEmbeddings = {};
        self.groupLabels = {};
        self.GNNEncoder.eval();
        with torch.no_grad():
            for subject in dataloader:
                batch = Batch.from_data_list(subject['graphs']);
                # GNNEncoder.forward does data.edge_attr.unsqueeze(-1) internally so
                # if edge_attr arrives as [E, 1] from preprocessBOLD, this will make it [E, 1, 1] and it will crash
                # we need to verify edge_attr shape before running so could u either remove unsqueeze in encoder or ensure
                # preprocessBOLD saves edge_attr as [E] not [E, 1].
                embeddings = self.GNNEncoder(batch);
                subjectId = subject['subject_id'];
                self.subjectEmbeddings[subjectId] = embeddings;
                self.groupLabels[subjectId] = int(subject['group_label'].view(-1)[0].item());

    def setAttention(self, attentionModel):
        self.attentionEmbeddings = {};
        self.attentionWeights = {};
        attentionModel.eval();
        with torch.no_grad():
            for subjectId, embeddings in self.subjectEmbeddings.items():
                output, weights, tau = attentionModel(embeddings);
                self.attentionEmbeddings[subjectId] = output;
                self.attentionWeights[subjectId] = weights;
        self.tau = tau.item();
        return self.attentionEmbeddings;

    def _stack(self, embDict):
        ids = list(embDict.keys());
        tensor = torch.stack([embDict[i] for i in ids]);
        return tensor, ids;

    def _split_fm_hc(self):
        self.fmEmbed = {};
        self.hcEmbed = {};
        for subjectId, embedding in self.attentionEmbeddings.items():
            if self.groupLabels[subjectId] == 0:
                self.fmEmbed[subjectId] = embedding;
            elif self.groupLabels[subjectId] == 1:
                self.hcEmbed[subjectId] = embedding;

    def validate_hc_sep(self):
        ids = list(self.attentionEmbeddings.keys());
        embeddings = torch.stack([self.attentionEmbeddings[i] for i in ids]).detach().cpu().numpy();
        labels = np.array([self.groupLabels[i] for i in ids]);
        self.hcSepSilh = silhouette_score(embeddings, labels);
        self.hcSepPermP = cluster_evaluate().perm(embeddings, labels);
        return self.hcSepSilh;

    def project_ortho(self, fmEmbeddings, hcEmbeddings):
        fmCentroid = fmEmbeddings.mean(dim=0);
        hcCentroid = hcEmbeddings.mean(dim=0);
        vSep = fmCentroid - hcCentroid;
        vSep = vSep / vSep.norm();
        projCoeff = fmEmbeddings @ vSep;
        zPerp = fmEmbeddings - torch.outer(projCoeff, vSep);
        self.hcC = hcCentroid - (hcCentroid @ vSep) * vSep;
        return zPerp;

    def compute_centroid_distances(self, fmEmbeddings, labels, hcCentroid):
        distances = [];
        for subtype in sorted(set(labels)):
            mask = torch.tensor(labels == subtype);
            subtypeCentroid = fmEmbeddings[mask].mean(dim=0);
            distances.append((subtypeCentroid - hcCentroid).norm().item());
        return np.array(distances);  #caller stores so it can be called per-space

    def KMeansUse(self, takeTensor = None, subjectIds = None, skip_perm = False, skip_gap = False):
        if takeTensor is None:
            self._split_fm_hc();
            takeTensor, subjectIds = self._stack(self.fmEmbed);
        takeTensor = takeTensor.detach().cpu().numpy();
        takeTensor = takeTensor / (np.linalg.norm(takeTensor, axis=1, keepdims=True) + 1e-8);
        numSubjects = takeTensor.shape[0];
        minClusterSize = max(config.minClusterSizeFloor, round(config.minClusterSizeFraction * numSubjects));
        trialSave = [];
        labelSave = [];
        members = [];
        for k in config.kmeansKRange:
            meaner = KMeans(n_clusters = k, n_init = config.kmeansNInit, random_state = config.randomSeed);
            labels = meaner.fit_predict(takeTensor);
            members.append(np.bincount(labels).min());
            score = silhouette_score(takeTensor, labels);
            trialSave.append(score);
            labelSave.append(labels);
        boolList = [None for _ in range(0,len(labelSave))];
        for i in range(0,len(labelSave)):
            if members[i]<minClusterSize:
                boolList[i] = False;
            else:
                boolList[i] = True;
        passedSave = [];
        for i in range(0,len(boolList)):
            if boolList[i] ==True:
                passedSave.append(trialSave[i]);
        passedIdx = [index for index in range(0,len(boolList)) if boolList[index]];
        if not passedIdx:
            bestIdx = trialSave.index(max(trialSave));
            sizeOk = False;
        else:
            bestIdx = max(passedIdx, key=lambda i : trialSave[i]);
            sizeOk = True;
        bestLabels = labelSave[bestIdx];
        evaluator = cluster_evaluate();
        kSil = config.kmeansKRange[bestIdx];  #silhouette picks final k
        if skip_gap: 
            gapDict = {kk: {"gap": np.nan, "s": np.nan} for kk in config.kmeansKRange};
            kGap = np.nan;
        else:
            gapDict = evaluator.gap_stat(takeTensor, k=config.kmeansKRange);  
            kGap = evaluator.gap_k(gapDict, config.kmeansKRange);  
        if skip_perm: 
            permP = np.nan;
        else:
            permP = evaluator.perm(takeTensor, bestLabels);
        permColumn = [np.nan for _ in config.kmeansKRange]
        permColumn[bestIdx] = permP;
        trialFrame = pd.DataFrame({"Subject_Id": subjectIds, "Label": bestLabels});
        trialSave = pd.DataFrame({"k": config.kmeansKRange, "silhouette_score": trialSave, "gap_stat": [gapDict[kk]["gap"] for kk in config.kmeansKRange], "gap_se": [gapDict[kk]["s"] for kk in config.kmeansKRange], "permutation_p": permColumn, "k_selected_silhouette": kSil, "k_selected_gap": kGap});
        return [trialSave, trialFrame, bestLabels, permP, sizeOk];

    def UMAPPING(self, array):
        array = array.detach().cpu().numpy();
        coords = umap.UMAP(n_neighbors = 10, min_dist = 0.1, random_state = config.randomSeed);
        return coords.fit_transform(array);

    def saveAll(self, cWeights, embeddingsAtt, kScore, labelFK, coords, orthoLabels, orthoScores):
        path = config.clusterOutput;
        path.mkdir(parents=True, exist_ok=True);
        labelFK.to_csv(path / "K-Means-Labeling.csv", index=False);
        np.save(path / "Embeddings.npy", embeddingsAtt.detach().cpu().numpy());
        kScore.to_csv(path / "silhouette-scores.csv");
        np.save(path / "UMAP-COORDS.npy", coords);
        np.save(path / "attentionWeights.npy", cWeights.detach().cpu().numpy());
        with open(path / "hc_separation_silhouette.json", "w") as f:
            json.dump({"hc_separation_silhouette": float(self.hcSepSilh), "permutation_p": float(self.hcSepPermP)}, f);  
        orthoLabels.to_csv(path / "orthogonal_labels.csv", index=False);
        orthoScores.to_csv(path / "orthogonal_silhouette_scores.csv");
        np.save(path / "centroid_distances.npy", self.centroidDistances);
        np.save(path / "centroid_distances_unprojected.npy", self.hcCUnprojDistances);
        with open(path / "tau_value.txt", "w") as f:
            f.write(str(self.tau));

    def clusterEX(self, dataloader, attentionPooler):
        self.deploy(dataloader);
        self.setAttention(attentionPooler);
        self._split_fm_hc();
        self.validate_hc_sep();
        bestTrial = self.KMeansUse();
        self.fmClusterPermP=bestTrial[3]  # perm p for ORIGINAL FM-only
        fmTensor, fmIds = self._stack(self.fmEmbed);
        hcTensor, hcIds = self._stack(self.hcEmbed);
        fmProjected = self.project_ortho(fmTensor, hcTensor);
        orthoTrial = self.KMeansUse(fmProjected, fmIds);
        self.orthoClusterPermP=orthoTrial[3]  # perm p for ORTHOGONAL-PROJECTED
        self.centroidDistances = self.compute_centroid_distances(fmProjected, orthoTrial[2], self.hcC);  # projected
        self.hcCUnprojDistances = self.compute_centroid_distances(fmTensor, bestTrial[2], hcTensor.mean(dim=0));  # unproj severity continuum
        coordinateVisuals = self.UMAPPING(fmTensor);
        weightMatrix = torch.stack([self.attentionWeights[i] for i in fmIds]);
        self.saveAll(weightMatrix, fmTensor, bestTrial[0], bestTrial[1], coordinateVisuals, orthoTrial[1], orthoTrial[0]);

if __name__ == "__main__":
    from gnn_encoder import GNNEncoder;
    from models.attention_pool import condition_attention_pool;
    from torch.utils.data import DataLoader;
    conditionList = ["Neutral - OBSERVAR", "Negativo - OBSERVAR", "Negativo - REDUCIR", "Negativo - SUPRIMIR", "Happy - OBSERVAR", "Happy - SUPRIMIR", "Happy - INCREMENTAR"];  # paper events.tsv order
    dataset = datasetPreparation(fm_only=False);
    dataList = dataset.subjectList;
    data = dataset.subjectData; 
    attention = condition_attention_pool(d_model=config.dModel, num_cons=config.nConditions);  #no hardcode
    encoder = GNNEncoder();
    checkpoint = torch.load(config.trainSave, map_location='cpu');
    encoder.load_state_dict(checkpoint['model']);
    attention.load_state_dict(checkpoint['pool']);
    runCluster = cluster(encoder, config.trainSave, conditionList, dataList);
    runCluster.clusterEX(data, attention);
