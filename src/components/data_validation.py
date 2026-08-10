import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.constants import CONFIG_FILE_PATH
from src.exception import CustomException
from src.logger import logger
from src.utils.common import read_yaml


@dataclass
class DataValidationConfig:
    root_dir: str
    schema_file_path: str
    status_file_path: str


class DataValidation:

    def __init__(self):

        try:
            config = read_yaml(CONFIG_FILE_PATH)

            validation_config = config["data_validation"]

            self.validation_config = DataValidationConfig(
                root_dir=validation_config["root_dir"],
                schema_file_path=validation_config[
                    "schema_file_path"
                ],
                status_file_path=validation_config[
                    "status_file_path"
                ],
            )

        except Exception as e:
            raise CustomException(e, sys)

    def _write_status(self, status):

        try:

            os.makedirs(
                self.validation_config.root_dir,
                exist_ok=True
            )

            with open(
                self.validation_config.status_file_path,
                "w"
            ) as file:

                file.write(
                    f"Validation status: {status}"
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

            actual_columns = list(
                df.columns
            )

            if actual_columns != expected_columns:

                logger.error(
                    f"Column validation failed.\n"
                    f"Expected: {expected_columns}\n"
                    f"Actual: {actual_columns}"
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

            for column, expected_type in (
                expected_types.items()
            ):

                actual_type = str(
                    df[column].dtype
                )

                if actual_type != expected_type:

                    logger.error(
                        f"Invalid data type for "
                        f"{column}. "
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

            # Load data
            df = pd.read_csv(
                data_path
            )

            logger.info(
                f"Dataset loaded. Shape: {df.shape}"
            )

            # Column validation
            if not self.validate_columns(df):

                self._write_status(False)

                return False

            # Data type validation
            if not self.validate_data_types(df):

                self._write_status(False)

                return False

            # Missing values
            null_count = (
                df.isnull()
                .sum()
                .sum()
            )

            if null_count > 0:

                logger.error(
                    f"Missing values found: "
                    f"{null_count}"
                )

                self._write_status(False)

                return False

            logger.info(
                "Missing value validation passed."
            )

            # Duplicate rows
            duplicate_count = (
                df.duplicated()
                .sum()
            )

            if duplicate_count > 0:

                logger.error(
                    f"Duplicate rows found: "
                    f"{duplicate_count}"
                )

                self._write_status(False)

                return False

            logger.info(
                "Duplicate validation passed."
            )

            # Validation successful
            self._write_status(True)

            logger.info(
                "Data Validation completed successfully."
            )

            return True

        except Exception as e:

            logger.exception(
                "Error occurred during Data Validation."
            )

            self._write_status(False)

            raise CustomException(e, sys)


if __name__ == "__main__":

    validator = DataValidation()

    result = validator.validate_data(
        "artifacts/data_ingestion/train.csv"
    )

    print(
        "\nValidation Result:",
        result
    )