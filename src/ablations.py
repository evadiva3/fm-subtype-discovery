import torch;
import pandas as pd;
import numpy as np;
import os;
class ablate(trainer):
    def __init__(self, model, loss, optimize, schedule, device, dire):
        super().__init__(model, loss, optimize, schedule, device, dire);
        self.classificationHead = nn.Linear(64,2);

    def train_epoch(self, dataloader, augmentor = None):
        self.model.train();
        loss = torch.nn.CrossEntropyLoss();
        i=0;
        total_loss=0;
        for batch in dataloader:
            batch=batch.to(self.device);
            self.optimize.zero_grad();
            embedding=self.model(batch);
            predictions = self.classificationHead(embedding);
            output = loss(predictions, batch.y.squeeze(1));
            output.backward();
            self.optimize.step();
            total_loss+=output.item();
            i+=1;
        self.schedule.step();
        return(total_loss/i);
def ablate2(encoder, trainSplit, testSplit, loss, optimizer, device, direct, augmentor):
    from train import trainer;
    run = trainer(encoder, loss, optimizer, scheduler, device, direct);
    run.fit(trainSplit, testSplit, augmentor);
    #run clustering that skips attention pooling and outputs to kmeans directly
def ablate3(encoder, loss, device, augmentor, direct):
    from train import trainer;
    from dataset import datasetPreparation;
    from torch_geometric.data import DataLoader;
    from torch.utils.data import random_split;
    from torch.optim import AdamW;
    from transformers import get_cosine_schedule_with_warmup;
    dataList = datasetPreparation(False);
    data = dataList.DataList;
    torch.save(encoder.state_dict(), "tempModelState.pt");
    for i in range(0,7):
        encoder.load_state_dict(torch.load("tempModelState.pt"));
        optimizer = AdamW(encoder.parameters(), lr=1e-4,weight_decay=1e-4);
        name = f"Condition{i}";
        condition = [];
        for instance in data:
            if instance.condition==dataList.conditionNames[i]:
                condition.append(instance);
        trainSplit, testSplit = random_split(condition, [0.8,0.2]);
        trainLoad = DataLoader(trainSplit, batch_size=8, shuffle=True);
        testLoad = DataLoader(testSplit, batch_size=8, shuffle=True);
        stepSize = len(trainLoad);
        epochs = 200;
        totalSteps = stepSize * epochs;
        partitionWarmupSteps = int(totalSteps * .10);
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = partitionWarmupSteps, num_training_steps=totalSteps);
        run = trainer(encoder, loss, optimizer, scheduler, device,os.path.join(direct,name));
        run.fit(trainLoad, testLoad, augmentor);
        # needs to run clustering
    os.remove("tempModelState.pt");
def ablate4(FCMatrix, folder : str, conditionList, saveFolder, checkpointName):
    import pathlib as Path;
    from sklearn.decomposition import PCA;
    from clustering import cluster;
    dataFolderPath = Path(folder);
    stackFCVec = [];
    subjectList = [];
    pca = PCA(n_components=64);
    for sublist in dataFolderPath.iterdir():
        cFCL = [];
        subjectList.append(sublist.name);
        for i in range(0,7)
            cFCL.append(np.load(self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_FCMatrixCondition" + conditionList[i].replace(" ", "") + ".npy"));
        cFCL = np.array(cFCL);
        cFCL = np.mean(cFCL, axis=0);
        rowInd, colInd = np.triu_indices(200, k=1);
        stackFCVec.append(cFCL[rowInd, colInd]);
    vecAVGCON = np.stack(stackFCVec, axis = 0);
    output = pca.fit_transform(vecAVGCON);
    kmeans = cluster(None, folder, checkpointName, conditionList, None);
    kTrials = kmeans.KMeansUse(output, subjectList);
    kTrials[0].to_csv(dataFolderPath/saveFolder/"silhouette-scores.csv");
    kTrials[1].to_csv(dataFolderPath/saveFolder/"K-Means-Labeling.csv");
        
