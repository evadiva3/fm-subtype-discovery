from pathlib import Path
from config import config
def test_repo_relative_data_root_defaults_to_workspace_data_dir():
    e=(Path(__file__).resolve().parents[1]/"data").resolve();assert config.dataRoot==e;assert config.subjectDataFolder==e/"Subjects";assert config.clinicalXlsx==e/"Clinical_fm_66.xlsx"
