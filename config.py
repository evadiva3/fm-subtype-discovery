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

    @property
    def CLINICALXLSX(self): return self.DATA_ROOT / "Clinical_fm_66.xlsx"

    @property
    def CLINICALCSV(self): return self.DATA_ROOT / "clinical_clean.csv"

    @property
    def SUBJECTDATAFOLDER(self): return self.DATA_ROOT / "Subjects"

    @property
    def MASKER(self): return str(self.DATA_ROOT / "schaefer200MNI.nii.gz")

    # Pipeline constants
    CONDITIONS = [
        "Neutral - OBSERVAR", "Negativo - OBSERVAR", "Happy - OBSERVAR",
        "Negativo - REDUCIR", "Negativo - SUPRIMIR", "Happy - SUPRIMIR",
        "Happy - INCREMENTAR"
    ]
    N_CONDITIONS=7
    N_NODES=200
    TR=2.0

    # Confound regressors 
    CONFOUND_COLUMNS = [
        "global_signal", "white_matter", "csf",
        "trans_x", "trans_x_derivative1", "trans_x_derivative1_power2", "trans_x_power2",
        "trans_y", "trans_y_derivative1", "trans_y_power2", "trans_y_derivative1_power2",
        "trans_z", "trans_z_derivative1", "trans_z_derivative1_power2", "trans_z_power2",
        "rot_x", "rot_x_derivative1", "rot_x_power2", "rot_x_derivative1_power2",
        "rot_y", "rot_y_derivative1", "rot_y_power2", "rot_y_derivative1_power2",
        "rot_z", "rot_z_derivative1", "rot_z_power2", "rot_z_derivative1_power2"
    ]

    # Model hyperparameters
    D_MODEL=64
    HEADS=4
    OUTPUT = 32;
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
    BOOTSTRAP_N_RESAMPLES=1000
    RANDOM_SEED=42
    BATCH_SIZE=8
    VAL_FRACTION=0.15

    # Clustering
    KMEANS_K_RANGE=[2,3,4]
    KMEANS_N_INIT=20

    # Motion / QC (verify_setup.py)
    FD_THRESHOLD=0.5
    FD_FRACTION_THRESHOLD=0.25

    # Statistics
    FDR_ALPHA=0.05

    # Baselines
    PCA_COMPONENTS=50
    GROUP_ICA_COMPONENTS=20
    SVM_CV_FOLDS=5

    # Graph construction 
    EDGE_PERCENTILE=80
    EDGE_PERCENTILE_SENSITIVITY=[75,80,85,90]

    # Device
    @property
    def DEVICE(self):
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")


config=Config()
