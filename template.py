import os
from pathlib import Path

project_name = "KitchenCast"

files = [
    ".github/workflows/.gitkeep",

    "config/config.yaml",
    "config/params.yaml",
    "config/schema.yaml",

    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    "notebooks/.gitkeep",

    "src/__init__.py",

    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_validation.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",

    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",

    "src/utils/__init__.py",
    "src/utils/common.py",

    "src/logger.py",
    "src/exception.py",
    "src/constants.py",

    "api/app.py",

    "tests/__init__.py",

    "artifacts/.gitkeep",

    "requirements.txt",
    "setup.py",
    "main.py",
    "README.md",
    "Dockerfile",
    ".gitignore",
]

for filepath in files:
    filepath = Path(filepath)

    # Create parent directories
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create file if it doesn't exist
    if not filepath.exists():
        filepath.touch()
        print(f"Created: {filepath}")
    else:
        print(f"Already exists: {filepath}")

print("\n KitchenCast project structure created successfully.")