import sys
from dataclasses import dataclass

import pandas as pd

from src.constants import CONFIG_FILE_PATH
from src.exception import CustomException
from src.logger import logger
from src.utils.common import read_yaml


@dataclass
class DataValidationConfig:
    schema_file_path: str


class DataValidation:

    def __init__(self):
        try:
            config = read_yaml(CONFIG_FILE_PATH)

            self.validation_config = DataValidationConfig(
                schema_file_path=config["data_validation"][
                    "schema_file_path"
                ]
            )

        except Exception as e:
            raise CustomException(e, sys)

    def validate_columns(self, df):

        try:
            schema = read_yaml(
                self.validation_config.schema_file_path
            )

            expected_columns = list(
                schema["COLUMNS"].keys()
            )

            actual_columns = list(df.columns)

            if actual_columns != expected_columns:
                logger.error(
                    "Column validation failed."
                )
                return False

            logger.info(
                "Column validation passed."
            )

            return True

        except Exception as e:
            raise CustomException(e, sys)

    def validate_data_types(self, df):

        try:
            schema = read_yaml(
                self.validation_config.schema_file_path
            )

            expected_types = schema["COLUMNS"]

            for column, expected_type in expected_types.items():

                actual_type = str(df[column].dtype)

                if actual_type != expected_type:

                    logger.error(
                        f"Invalid data type for {column}. "
                        f"Expected: {expected_type}, "
                        f"Got: {actual_type}"
                    )

                    return False

            logger.info(
                "Data type validation passed."
            )

            return True

        except Exception as e:
            raise CustomException(e, sys)

    def validate_data(self, data_path):

        try:
            logger.info(
                "Data Validation started."
            )

            df = pd.read_csv(data_path)

            if not self.validate_columns(df):
                return False

            if not self.validate_data_types(df):
                return False

            if df.isnull().sum().sum() > 0:

                logger.error(
                    "Missing values found."
                )

                return False

            if df.duplicated().sum() > 0:

                logger.error(
                    "Duplicate rows found."
                )

                return False

            logger.info(
                "Data Validation completed successfully."
            )

            return True

        except Exception as e:

            logger.exception(
                "Error occurred during Data Validation."
            )

            raise CustomException(e, sys)


if __name__ == "__main__":

    validator = DataValidation()

    result = validator.validate_data(
        "artifacts/data_ingestion/train.csv"
    )

    print(
        "Validation Result:",
        result
    )