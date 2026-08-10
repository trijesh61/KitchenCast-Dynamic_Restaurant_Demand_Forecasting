import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.xgboost
import numpy as np
import joblib

from scipy.sparse import load_npz

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor

from src.constants import CONFIG_FILE_PATH
from src.exception import CustomException
from src.logger import logger
from src.utils.common import read_yaml


@dataclass
class ModelTrainerConfig:
    root_dir: str
    model_path: str
    target_column: str


class ModelTrainer:

    def __init__(self):

        try:

            config = read_yaml(
                CONFIG_FILE_PATH
            )

            trainer_config = config[
                "model_trainer"
            ]

            self.trainer_config = ModelTrainerConfig(
                root_dir=trainer_config["root_dir"],
                model_path=trainer_config["model_path"],
                target_column=trainer_config["target_column"],
            )

        except Exception as e:

            raise CustomException(e, sys)

    def initiate_model_training(
        self,
        train_features_path,
        test_features_path,
        train_target_path,
        test_target_path,
    ):

        try:

            logger.info(
                "Model Training started."
            )

            # ----------------------------------------
            # Load transformed features
            # ----------------------------------------

            X_train = load_npz(
                train_features_path
            )

            X_test = load_npz(
                test_features_path
            )

            # ----------------------------------------
            # Load targets
            # ----------------------------------------

            y_train = np.load(
                train_target_path
            )

            y_test = np.load(
                test_target_path
            )

            logger.info(
                f"X_train shape: {X_train.shape}"
            )

            logger.info(
                f"X_test shape: {X_test.shape}"
            )

            logger.info(
                f"y_train shape: {y_train.shape}"
            )

            logger.info(
                f"y_test shape: {y_test.shape}"
            )

            # ----------------------------------------
            # Create model
            # ----------------------------------------

            model_params = {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": -1,
            }

            model = XGBRegressor(
                **model_params
            )

            # ----------------------------------------
            # MLflow
            # ----------------------------------------

            mlflow.set_experiment(
                "KitchenCast-Demand-Forecasting"
            )

            with mlflow.start_run():

                logger.info(
                    "Training XGBoost model."
                )

                model.fit(
                    X_train,
                    y_train
                )

                # ----------------------------------------
                # Predictions
                # ----------------------------------------

                y_pred = model.predict(
                    X_test
                )

                # ----------------------------------------
                # Metrics
                # ----------------------------------------

                mae = mean_absolute_error(
                    y_test,
                    y_pred
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        y_pred
                    )
                )

                r2 = r2_score(
                    y_test,
                    y_pred
                )

                logger.info(
                    f"MAE: {mae:.4f}"
                )

                logger.info(
                    f"RMSE: {rmse:.4f}"
                )

                logger.info(
                    f"R2: {r2:.4f}"
                )

                # ----------------------------------------
                # Log parameters
                # ----------------------------------------

                mlflow.log_params(
                    model_params
                )

                # ----------------------------------------
                # Log metrics
                # ----------------------------------------

                mlflow.log_metrics(
                    {
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2,
                    }
                )

                # ----------------------------------------
                # Log model
                # ----------------------------------------

                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model",
                )

                logger.info(
                    "Model logged to MLflow."
                )

            # ----------------------------------------
            # Save local model
            # ----------------------------------------

            os.makedirs(
                self.trainer_config.root_dir,
                exist_ok=True
            )

            joblib.dump(
                model,
                self.trainer_config.model_path
            )

            logger.info(
                f"Model saved at: "
                f"{self.trainer_config.model_path}"
            )

            logger.info(
                "Model Training completed successfully."
            )

            return (
                self.trainer_config.model_path
            )

        except Exception as e:

            logger.exception(
                "Error occurred during Model Training."
            )

            raise CustomException(e, sys)


if __name__ == "__main__":

    trainer = ModelTrainer()

    model_path = trainer.initiate_model_training(

        "artifacts/data_transformation/X_train.npz",

        "artifacts/data_transformation/X_test.npz",

        "artifacts/data_transformation/y_train.npy",

        "artifacts/data_transformation/y_test.npy",
    )

    print(
        "\nModel saved at:",
        model_path
    )