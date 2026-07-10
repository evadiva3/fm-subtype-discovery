from ray import tune, train as rayTrain;
import torch;
def runscript(config1):
    from config import config;
    from train import trainer;
    from gnn_encoder import GNNEncoder
    from contrastive_loss import NTXentLoss
    from attention_pool import condition_attention_pool as conditionAttentionPool
    from augmentations import graph_augmentor as graphAugmentor;
    from dataset import datasetPreparation
    from torch.utils.data import DataLoader, random_split as randomSplit;
    from train import joint_train as jointTrain;
    dataset = datasetPreparation();
    class GroupedWrapper(torch.utils.data.Dataset):
        def __init__(self, subject_data):
            self.subject_data=subject_data
        def __len__(self):
            return len(self.subject_data)
        def __getitem__(self, idx):
            return self.subject_data[idx]
    direct = rayTrain.get_context().get_trial_dir();
    config.dModel = config1["dModel"];
    config.heads = config1["heads"];
    config.output = config1["output"];
    config.layers = config1["layers"];
    config.dropout = config1["dropout"];
    config.lr = config1["lr"];
    config.weightDecay = config1["weightDecay"];
    config.maskRate = config1["maskRate"];
    config.ntXentTemp = config1["ntXentTemp"];
    config.batchSize = config1["batchSize"];
    data = DataLoader(GroupedWrapper(dataset.subjectData))
    NTXLoss = NTXentLoss();
    encoder = GNNEncoder();
    attentionPooling = conditionAttentionPool();
    augmentation = graphAugmentor();
    splitGeneration = torch.Generator().manual_seed(config.randomSeed);
    size = len(data);
    partial = int(size*config.valFraction);
    split = size-partial;
    train, test = randomSplit(data, [split, partial], generator = splitGeneration);
    training = DataLoader(train, batch_size=config.batchSize, shuffle = True, collate_fn = lambda b:b);
    testing = DataLoader(test, batch_size = config.batchSize, shuffle = False, collate_fn=lambda b:b);
    jointTrain(encoder, attentionPooling, NTXLoss, training, testing, augmentation, config.device, direct,config.tuneEpochs, None, None, None);
searchSpace = {"dModel": tune.randint(16, 129), "heads": tune.randint(2,9), "output": tune.randint(8,65), "layers": tune.randint(2,5), "dropout": tune.uniform(0.0, 0.3), "lr": tune.loguniform(1e-5,1e-3), "weightDecay": tune.loguniform(1e-4, 1e-1), "maskRate": tune.uniform(0.05, 0.25), "nXentTemp": tune.uniform(0.2, 1.0), "batchSize": tune.randint(4, 24)}