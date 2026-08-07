import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    artifact_dir: str = os.path.join("artifacts", "data_ingestion")
    raw_data_path: str = os.path.join(artifact_dir, "raw.csv")
    train_data_path: str = os.path.join(artifact_dir, "train.csv")
    test_data_path: str = os.path.join(artifact_dir, "test.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Data Ingestion started.")

        try:
            source_data_path = os.path.join("data", "raw", "train.csv")

            logger.info(f"Reading dataset from: {source_data_path}")
            df = pd.read_csv(source_data_path)

            logger.info("Creating artifacts directory.")
            os.makedirs(self.ingestion_config.artifact_dir, exist_ok=True)

            logger.info("Saving raw dataset.")
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logger.info("Performing train-test split.")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
            )

            logger.info("Saving train and test datasets.")
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
            )

            logger.info("Data Ingestion completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logger.exception("Error occurred during Data Ingestion.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    print("Train File:", train_path)
    print("Test File:", test_path)