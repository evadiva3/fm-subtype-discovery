import os;
import torch;
import pandas as pd;
import numpy as np;
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score;
import umap
from pathlib import Path;
from torch.utils.data import Dataset
from dataset import datasetPreparation;
class cluster():
    def __init__(self, GNNEncoder, Directory, checkpointName, conditionList, subjectList):
        self.GNNEncoder = GNNEncoder;
        self.PTPath = os.path.join(Directory, checkpointName);
        self.subjectDList = [[None for _ in range(0, len(conditionList))] for _ in range (0, len(subjectList))];
        self.conditionList = conditionList;
        self.subjectList = subjectList;
    def deploy(self,dataset):
        self.GNNEncoder.load_state_dict(torch.load(self.PTPath));
        self.GNNEncoder.eval();
        with torch.no_grad():
            for batch in dataset:
                embeddings = self.GNNEncoder(batch);
                self.organizeEmbeddings(embeddings, batch);
    def organizeEmbeddings(self,batchTense, batch):
        for i in range(len(batchTense)):
            self.subjectDList[self.subjectList.index(batch.subjectID[i])][self.conditionList.index(batch.condition[i])] = batchTense[i];
    def setAttention(self,attentionModel):
        masterAttList = [];
        masterWeightList = [];
        attentionModel.eval();
        with torch.no_grad():
            for subject in self.subjectDList:
                tensorMatrix = torch.stack(subject);
                output, weights = attentionModel(tensorMatrix);
                masterAttList.append(output);
                masterWeightList.append(weights);
        masterAttList = torch.stack(masterAttList);
        masterWeightList = torch.stack(masterWeightList);
        return (masterAttList, masterWeightList);
    def KMeansUse(self,takeTensor):
        takeTensor = takeTensor.numpy();
        trialSave = [];
        labelSave = [];
        k=2;
        for _ in range(0,3):
            meaner = KMeans(n_clusters = k, n_init=20, random_state=42);
            labels = meaner.fit_predict(takeTensor);
            score = silhouette_score(takeTensor, labels);
            trialSave.append(score);
            labelSave.append(labels);
            k+=1;
        labelSave = labelSave[trialSave.index(max(trialSave))];
        trialFrame = pd.DataFrame({"Subject_Id": self.subjectList, "Label":labelSave});
        trialSave = pd.DataFrame({"k": [2, 3, 4], "silhouette_score": trialSave});
        return [trialSave, trialFrame];
    def UMAPPING(self,array):
        array = array.numpy();
        coords = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42);
        return coords.fit_transform(array);
    def saveAll(self,CWeights, embeddingsAtt, KScore, labelfK, coords):
        path = Path("../ClusterResults");
        path.mkdir(parents=True, exist_ok=True);
        labelfK.to_csv(path/"K-Means-Labeling.csv", index=False);
        np.save(path/"Embeddings.npy",embeddingsAtt.numpy());
        KScore.to_csv(path/"silhouette-scores.csv");
        np.save(path/"UMAP-COORDS.npy",coords);
        np.save(path/"attentionWeights.npy",CWeights.numpy());

    def clusterEX(self,dataloader, attentionPooler):
        self.deploy(dataloader);
        attentionList, weightList = self.setAttention(attentionPooler);
        bestTrial = self.KMeansUse(attentionList);
        coordinateVisuals = self.UMAPPING(attentionList);
        self.saveAll(weightList, attentionList, bestTrial[0], bestTrial[1],coordinateVisuals);
if __name__ == "__main__":
    from gnn_encoder import GNNEncoder
    from models.attention_pool import condition_attention_pool
    from torch_geometric.data import DataLoader

    conditionList = ["Neutral - OBSERVAR", "Negativo - OBSERVAR", "Happy - OBSERVAR", 
                     "Negativo - REDUCIR", "Negativo - SUPRIMIR", "Happy - SUPRIMIR", 
                     "Happy - INCREMENTAR"]
    
    dataset = datasetPreparation()
    dataList = dataset.subjectList;
    data = DataLoader(dataset, batch_size=8, shuffle=False)
    attention = condition_attention_pool(d_model=64, num_cons=7);
    encoder = GNNEncoder();
    runCluster = cluster(encoder, "results/checkpoints", "best_model.pt", conditionList, dataList);
    runCluster.clusterEX(data, attention);
