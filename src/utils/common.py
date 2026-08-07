import os
import yaml
import joblib


def read_yaml(path_to_yaml):
    with open(path_to_yaml, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def create_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path):
    return joblib.load(file_path)