"""
Every module imports the shared instance:
    from config import config
"""

import os
import torch
from pathlib import Path


class Config:
    REPO_ROOT=Path(__file__).parent.resolve()
    DATA_ROOT=Path(os.environ.get("FM_DATA_ROOT", "../pathname"))
    RESULTS_ROOT=Path(os.environ.get("FM_RESULTS_ROOT", REPO_ROOT / "results"))

    @property
    def FC_MATRIX_DIR(self): return self.DATA_ROOT / "fc_matrices"

    @property
    def ROI_TIMESERIES_DIR(self): return self.DATA_ROOT / "roi_timeseries"

    @property
    def CHECKPOINT_DIR(self): return self.RESULTS_ROOT / "checkpoints"

    @property
    def FIGURES_DIR(self): return self.RESULTS_ROOT / "figures"

    @property
    def CLINICAL_CSV_PATH(self): return self.DATA_ROOT / "clinical_clean.csv"

    @property
    def JOINT_CHECKPOINT_PATH(self): return self.CHECKPOINT_DIR / "best_joint_model.pt"

    # Pipeline constants
    CONDITIONS = [
        "Neutral - OBSERVAR", "Negativo - OBSERVAR", "Happy - OBSERVAR",
        "Negativo - REDUCIR", "Negativo - SUPRIMIR", "Happy - SUPRIMIR",
        "Happy - INCREMENTAR"
    ]
    N_CONDITIONS=7
    N_NODES=200
    TR=2.0
    #  edge-density value.
    TOP_K_SENSITIVITY=[20, 30, 40, 50]

    # Model hyperparameters
    D_MODEL=64
    HEADS=4
    LAYERS=3
    DROPOUT=0.1
    LR=1e-4
    WEIGHT_DECAY=1e-2
    MASK_RATE=0.1
    NOISE_STD=0.1
    NT_XENT_TEMP=0.5

    # Training
    EPOCHS=200
    PATIENCE=10
    N_PERMUTATIONS=1000

    # Device 
    @property
    def DEVICE(self):
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    #Nikhil's stuff
    FCDATAFOLDER = "../data/FCData";
    SUBJECTDATAFOLDER = "../data/Subjects";
    MASKER = "../data/schaefer200MNI.nii.gz"
    CLINICALXLSX = "../data/Clinical_fm_66.xlsx";
    CLINICALCSV = "../data/Clinical_fm_66.csv";


config=Config()
