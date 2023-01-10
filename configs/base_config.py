from pathlib import Path
from typing import List
import os

class BaseSettings():
    TRAIN_DATA: Path = Path('data/train.json')
    TEST_DATA: Path = Path('data/private_test_labels.csv')
    TARGET_LABELS: List[str] = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
    TEST_LABELS: List[str] = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
    PROJECT_NAME: str = "mRNA-graph-models"
    EXPERIMENTS_FOLDER: Path = Path('experiments/models')

    @staticmethod
    def get_model_path(model_name: str) -> Path:
        model_path = BaseSettings.get_model_folder(model_name) / "model.pth"
        return model_path

    @staticmethod
    def get_model_folder(model_name: str) -> Path:
        model_path = BaseSettings.EXPERIMENTS_FOLDER / model_name

        # create folder if it doesn't exist
        if not model_path.exists():
            os.mkdir(model_path)

        return model_path

    # TODO get from env