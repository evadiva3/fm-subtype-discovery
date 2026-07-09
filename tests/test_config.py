from pathlib import Path

from config import config


def test_repo_relative_data_root_defaults_to_workspace_data_dir():
    expected = (Path(__file__).resolve().parents[1] / "data").resolve()
    assert config.dataRoot == expected
    assert config.subjectDataFolder == expected / "Subjects"
    assert config.clinicalXlsx == expected / "Clinical_fm_66.xlsx"
