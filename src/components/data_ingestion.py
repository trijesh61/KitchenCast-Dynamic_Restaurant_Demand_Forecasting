import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

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

            logger.info(
                f"Reading dataset from: "
                f"{self.ingestion_config.source_data_path}"
            )

            df = pd.read_csv(
                self.ingestion_config.source_data_path
            )

            logger.info(
                f"Dataset loaded successfully. Shape: {df.shape}"
            )

            logger.info("Creating data ingestion directory.")

            import os

            os.makedirs(
                self.ingestion_config.root_dir,
                exist_ok=True
            )

            logger.info("Saving raw dataset.")

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            logger.info("Performing train-test split.")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            logger.info("Saving train dataset.")

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            logger.info("Saving test dataset.")

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logger.info(
                "Data Ingestion completed successfully."
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
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

    print("Train File:", train_path)
    print("Test File:", test_path)