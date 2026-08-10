import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.constants import CONFIG_FILE_PATH
from src.exception import CustomException
from src.logger import logger
from src.utils.common import read_yaml


@dataclass
class DataIngestionConfig:
    root_dir: str
    source_data_path: str
    raw_data_path: str
    train_data_path: str
    test_data_path: str


class DataIngestion:

    def __init__(self):

        try:
            config = read_yaml(CONFIG_FILE_PATH)

            ingestion_config = config["data_ingestion"]

            self.ingestion_config = DataIngestionConfig(
                root_dir=ingestion_config["root_dir"],
                source_data_path=ingestion_config["source_data_path"],
                raw_data_path=ingestion_config["raw_data_path"],
                train_data_path=ingestion_config["train_data_path"],
                test_data_path=ingestion_config["test_data_path"],
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):

        try:

            logger.info("Data Ingestion started.")

            # Load source dataset
            logger.info(
                f"Reading dataset from: "
                f"{self.ingestion_config.source_data_path}"
            )

            df = pd.read_csv(
                self.ingestion_config.source_data_path
            )

            logger.info(
                f"Dataset loaded successfully. "
                f"Shape: {df.shape}"
            )

            # Sort chronologically
            if "week" not in df.columns:
                raise ValueError(
                    "'week' column is required for "
                    "chronological splitting."
                )

            df = df.sort_values(
                "week"
            ).reset_index(drop=True)

            # Create artifact directory
            os.makedirs(
                self.ingestion_config.root_dir,
                exist_ok=True
            )

            # Save raw copy
            logger.info("Saving raw dataset.")

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            # Chronological train-test split
            split_index = int(len(df) * 0.8)

            train_set = df.iloc[:split_index].copy()
            test_set = df.iloc[split_index:].copy()

            logger.info(
                f"Training data shape: {train_set.shape}"
            )

            logger.info(
                f"Testing data shape: {test_set.shape}"
            )

            # Save training data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            # Save testing data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logger.info(
                "Data Ingestion completed successfully."
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:

            logger.exception(
                "Error occurred during Data Ingestion."
            )

            raise CustomException(e, sys)


if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    print("\nData Ingestion Outputs:")
    print("Train:", train_path)
    print("Test :", test_path)