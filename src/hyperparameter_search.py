from ray import tune, train as rayTrain;
import torch;
import os;
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1";
from ray.tune.search.optuna import OptunaSearch;
from ray.tune.schedulers import ASHAScheduler;
import pandas as pd;
from config import config;
def runscript(config1, dataset):
    from config import config;
    import train;
    from gnn_encoder import GNNEncoder
    from contrastive_loss import NTXentLoss
    from attention_pool import condition_attention_pool as conditionAttentionPool
    from augmentations import graph_augmentor as graphAugmentor;
    from torch.utils.data import DataLoader, random_split as randomSplit;
    from train import joint_train as jointTrain;
    torch.manual_seed(config.randomSeed);
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.randomSeed);
    direct = tune.get_context().get_trial_dir();
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
    data = GroupedWrapper(dataset.subjectData);
    NTXLoss = NTXentLoss();
    encoder = GNNEncoder().to(config.device);
    attentionPooling = conditionAttentionPool().to(config.device);
    augmentation = graphAugmentor();
    splitGeneration = torch.Generator().manual_seed(config.randomSeed);
    size = len(data);
    partial = int(size*config.valFractionForTune);
    split = size-partial;
    train, test = randomSplit(data, [split, partial], generator = splitGeneration);
    dataset.normalizeData(train.indices);
    training = DataLoader(train, batch_size=config.batchSize, shuffle = True, collate_fn = lambda b:b, drop_last=True);
    testing = DataLoader(test, batch_size = config.batchSize, shuffle = False, collate_fn=lambda b:b);
    tuneName = os.path.basename(direct);
    encodeOut, attentionOut, trainLoss, valLoss = jointTrain(encoder, attentionPooling, NTXLoss, training, testing, augmentation, config.device, direct,config.tuneEpochs, None, None, None,tuneName);
class GroupedWrapper(torch.utils.data.Dataset):
        def __init__(self, subject_data):
            self.subject_data=subject_data
        def __len__(self):
            return len(self.subject_data)
        def __getitem__(self, idx):
            return self.subject_data[idx]
if __name__ == "__main__":
    from dataset import datasetPreparation;
    dataset = datasetPreparation();
    
    optuna = OptunaSearch(metric="valLoss", mode="min");
    ASHA = ASHAScheduler(time_attr="training_iteration", max_t=config.tuneEpochs, reduction_factor=2, grace_period=5, metric="valLoss", mode="min");
    searchSpace = {"dModel": tune.randint(16, 129), "heads": tune.randint(2,9), "output": tune.randint(8,65), "layers": tune.randint(2,5), "dropout": tune.uniform(0.0, 0.3), "lr": tune.loguniform(1e-5,3e-3), "weightDecay": tune.loguniform(1e-4, 1e-1), "maskRate": tune.uniform(0.05, 0.25), "ntXentTemp": tune.uniform(0.2, 1.0), "batchSize": tune.randint(8, 24)};
    rayTune = tune.Tuner(tune.with_parameters(runscript, dataset=dataset), param_space = searchSpace, tune_config=tune.TuneConfig(metric="valLoss",search_alg=optuna,mode="min", num_samples=config.sampleNum, max_concurrent_trials=config.maxConcurrents, scheduler=ASHA), run_config=tune.RunConfig(storage_path=config.rayStorage,failure_config=(tune.FailureConfig(max_failures=3))));
    result = rayTune.fit();
    pd.DataFrame([result.get_best_result(metric="valLoss", mode="min").config]).to_json(config.raySavePath);
