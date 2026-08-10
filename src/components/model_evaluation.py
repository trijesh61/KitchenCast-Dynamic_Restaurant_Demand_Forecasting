import json
import os
import sys
from dataclasses import dataclass

import joblib
import mlflow
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.constants import CONFIG_FILE_PATH
from src.exception import CustomException
from src.logger import logger
from src.utils.common import read_yaml


@dataclass
class ModelEvaluationConfig:
    root_dir: str
    metric_file_name: str
    mae_threshold: float


class ModelEvaluation:

    def __init__(self):

        try:
            config = read_yaml(
                CONFIG_FILE_PATH
            )

            evaluation_config = config[
                "model_evaluation"
            ]

            self.evaluation_config = (
                ModelEvaluationConfig(
                    root_dir=evaluation_config[
                        "root_dir"
                    ],
                    metric_file_name=evaluation_config[
                        "metric_file_name"
                    ],
                    mae_threshold=float(
                        evaluation_config[
                            "mae_threshold"
                        ]
                    ),
                )
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(
        self,
        model_path,
        test_features_path,
        test_target_path,
    ):

        try:

            logger.info(
                "Model Evaluation started."
            )

            # Load model
            model = joblib.load(
                model_path
            )

            # Load test data
            X_test = load_npz(
                test_features_path
            )

            y_test = np.load(
                test_target_path
            )

            logger.info(
                "Model and test data loaded."
            )

            # Generate predictions
            y_pred = model.predict(
                X_test
            )

            # Calculate metrics
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

            metrics = {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
            }

            logger.info(
                f"MAE: {mae:.4f}"
            )

            logger.info(
                f"RMSE: {rmse:.4f}"
            )

            logger.info(
                f"R2: {r2:.4f}"
            )

            # Create evaluation directory
            os.makedirs(
                self.evaluation_config.root_dir,
                exist_ok=True
            )

            # Save metrics
            metric_path = os.path.join(
                self.evaluation_config.root_dir,
                self.evaluation_config.metric_file_name,
            )

            with open(
                metric_path,
                "w"
            ) as file:

                json.dump(
                    metrics,
                    file,
                    indent=4
                )

            logger.info(
                f"Metrics saved to: {metric_path}"
            )

            # Model quality gate
            model_passed = (
                mae
                <= self.evaluation_config.mae_threshold
            )

            if model_passed:

                logger.info(
                    "Model passed the MAE threshold."
                )

            else:

                logger.error(
                    "Model failed the MAE threshold."
                )

            # Log evaluation metrics to MLflow
            with mlflow.start_run(
                run_name="model-evaluation"
            ):

                mlflow.log_metrics(
                    metrics
                )

                mlflow.log_param(
                    "mae_threshold",
                    self.evaluation_config.mae_threshold
                )

                mlflow.log_param(
                    "model_passed",
                    model_passed
                )

                mlflow.log_artifact(
                    metric_path
                )

            logger.info(
                "Model Evaluation completed."
            )

            return model_passed

        except Exception as e:

            logger.exception(
                "Error occurred during Model Evaluation."
            )

            raise CustomException(e, sys)


if __name__ == "__main__":

    evaluator = ModelEvaluation()

    result = evaluator.initiate_model_evaluation(

        "artifacts/model_trainer/model.pkl",

        "artifacts/data_transformation/X_test.npz",

        "artifacts/data_transformation/y_test.npy",
    )

    print(
        "\nModel Passed:",
        result
    )