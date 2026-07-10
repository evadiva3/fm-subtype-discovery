"""
Every module imports the shared instance:
    from config import config
"""

import os
import torch
from pathlib import Path
import pandas as pd;

class Config:
    repoRoot=Path(__file__).parent.resolve()
    dataRoot=Path(os.environ.get("FM_DATA_ROOT", repoRoot / "data"))
    resultsRoot=Path(os.environ.get("FM_RESULTS_ROOT", repoRoot / "results"))
    loadBestParams=dataRoot / "tune" / "bestParamTune.json"
    tuneParams=pd.read_json(loadBestParams)

    @property
    def checkpointDir(self): return self.resultsRoot / "checkpoints"

    @property
    def figuresDir(self): return self.resultsRoot / "figures"

    @property
    def jointCheckpointPath(self): return self.checkpointDir / "best_joint_model.pt"

    @property
    def exclusionManifestPath(self): return self.resultsRoot / "subject_exclusions.csv"

    @property
    def fcDataFolder(self): return self.dataRoot / "FCData"

    @property
    def clinicalXlsx(self): return self.dataRoot / "Clinical_fm_66.xlsx"

    @property
    def clinicalCsv(self): return self.dataRoot / "clinical_clean.csv"

    @property
    def subjectDataFolder(self): return self.dataRoot / "Subjects"

    @property
    def masker(self): return str(self.dataRoot / "schaefer200MNI.nii.gz")

    # Pipeline constants
    conditions = ["Neutral - OBSERVAR", "Negativo - OBSERVAR", "Negativo - REDUCIR", "Negativo - SUPRIMIR", "Happy - OBSERVAR", "Happy - SUPRIMIR", "Happy - INCREMENTAR"]  # paper events.tsv order
    nConditions=7
    nNodes=200
    tr=2.0
    highPassCutoff=0.01

    # Confound regressors
    confoundColumns = ["global_signal", "white_matter", "csf", "trans_x", "trans_x_derivative1", "trans_x_derivative1_power2", "trans_x_power2", "trans_y", "trans_y_derivative1", "trans_y_power2", "trans_y_derivative1_power2", "trans_z", "trans_z_derivative1", "trans_z_derivative1_power2", "trans_z_power2", "rot_x", "rot_x_derivative1", "rot_x_power2", "rot_x_derivative1_power2", "rot_y", "rot_y_derivative1", "rot_y_power2", "rot_y_derivative1_power2", "rot_z", "rot_z_derivative1", "rot_z_power2", "rot_z_derivative1_power2"]
    # Model hyperparameters
    dModel=int(tuneParams.at[0,"D_MODEL"])
    heads=int(tuneParams.at[0,"HEADS"])
    output=int(tuneParams.at[0,"OUTPUT"])
    layers=int(tuneParams.at[0,"LAYERS"])
    dropout=float(tuneParams.at[0,"DROPOUT"])
    lr=float(tuneParams.at[0,"LR"])
    weightDecay=float(tuneParams.at[0,"WEIGHT_DECAY"])
    maskRate=float(tuneParams.at[0,"MASK_RATE"])
    noiseStd=float(tuneParams.at[0,"NOISE_STD"])
    ntXentTemp=float(tuneParams.at[0,"NT_XENT_TEMP"])
    batchSize=int(tuneParams.at[0,"BATCH_SIZE"])
    # Augmentation apply probs (per view, independent of strength)
    maskApplyProb=0.5
    noiseApplyProb=0.5

   #Tuning Hypers:
    tuneEpochs = 100;
    # Training
    epochs=200
    patience=10
    nPermutations=1000
    bootstrapNResamples=1000
    randomSeed=42
    valFraction=0.15
    # Clustering
    kmeansKRange=[2,3,4]
    kmeansNInit=20

    # Motion / QC (verify_setup.py)
    fdThreshold=0.5
    fdFractionThreshold=0.25

    # Statistics
    fdrAlpha=0.05

    # Baselines
    pcaComponents=50
    groupIcaComponents=20
    svmCvFolds=5

    # Graph construction
    edgePercentile=80
    edgePercentileSensitivity=[75,80,85,90]
    
    #gap
    gapB=10

    # Device
    @property
    def device(self):
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")


config=Config()
