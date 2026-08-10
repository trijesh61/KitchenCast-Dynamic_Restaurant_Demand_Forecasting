import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.exception import CustomException
from src.logger import logger


class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            logger.info("Starting Data Ingestion.")

            data_ingestion = DataIngestion()

            train_path, test_path = (
                data_ingestion.initiate_data_ingestion()
            )

            logger.info(
                "Data Ingestion completed successfully."
            )

            return train_path, test_path

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self, train_path):
        try:
            logger.info("Starting Data Validation.")

            data_validation = DataValidation()

            validation_status = (
                data_validation.validate_data(train_path)
            )

            if not validation_status:
                raise Exception(
                    "Data Validation failed."
                )

            logger.info(
                "Data Validation completed successfully."
            )

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            train_path, test_path = (
                self.start_data_ingestion()
            )

            validation_status = (
                self.start_data_validation(train_path)
            )

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.run_pipeline()