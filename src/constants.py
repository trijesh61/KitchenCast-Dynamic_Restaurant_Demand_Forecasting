from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "config" / "params.yaml"
SCHEMA_FILE_PATH = ROOT_DIR / "config" / "schema.yaml"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"