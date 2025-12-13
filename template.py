import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "drift-monitoring-project"

list_of_files = [
    # Notebook
    "notebooks/eda.ipynb",
    "notebooks/02_train_model.ipynb",
    "notebooks/03_drift_reports.ipynb",

    # data sources
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/production_batches/.gitkeep",

    # artifacts
    "artifacts/models//.gitkeep",
    "artifacts/encoders/.gitkeep",
    "artifacts/reports/.gitkeep",

    # src files
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/model_trainer.py",
    "src/components/drift_monitor.py",

    # config files
    "src/configuration/config.yaml",
    "src/configuration//__init__.py",

    # constants
    "src/constants/__init__.py",
    "src/constants/constants.py",

    # pipeline files
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/monitoring_pipeline.py.py",

    # utils files
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/exception.py",
    "src/utils/common.py",

    "templates/index.html",

    "static",
    "mlruns",
    "app.py"

    "docker-compose.yml",
    "Dockerfile",
    "requirements.txt",
    "README.md",
    ".github/workflows/ci.yml"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory if not exists
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    # Create empty file if not exists OR file is empty
    if (not os.path.exists(filepath)) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")
