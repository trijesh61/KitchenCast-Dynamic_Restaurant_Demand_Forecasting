import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse, save_npz
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.constants import CONFIG_FILE_PATH
from src.exception import CustomException
from src.logger import logger
from src.utils.common import read_yaml


@dataclass
class DataTransformationConfig:
    root_dir: str
    preprocessor_path: str
    meal_info_path: str
    center_info_path: str


class DataTransformation:

    def __init__(self):

        try:
            config = read_yaml(CONFIG_FILE_PATH)

            transformation_config = config["data_transformation"]

            self.transformation_config = DataTransformationConfig(
                root_dir=transformation_config["root_dir"],
                preprocessor_path=transformation_config["preprocessor_path"],
                meal_info_path=transformation_config["meal_info_path"],
                center_info_path=transformation_config["center_info_path"],
            )

        except Exception as e:
            raise CustomException(e, sys)

    def _prepare_data(
        self,
        df,
        meal_df,
        center_df
    ):
        """
        Merge lookup tables and create features.
        """

        try:
            # Merge meal information
            df = df.merge(
                meal_df,
                on="meal_id",
                how="left"
            )

            # Merge fulfillment center information
            df = df.merge(
                center_df,
                on="center_id",
                how="left"
            )

            # Feature: absolute discount
            df["discount"] = (
                df["base_price"]
                - df["checkout_price"]
            )

            # Feature: discount percentage
            df["discount_pct"] = (
                (
                    df["base_price"]
                    - df["checkout_price"]
                )
                / df["base_price"]
            ) * 100

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:
            logger.info(
                "Data Transformation started."
            )

            # --------------------------------------------------
            # 1. Load datasets
            # --------------------------------------------------

            train_df = pd.read_csv(train_path)

            test_df = pd.read_csv(test_path)

            meal_df = pd.read_csv(
                self.transformation_config.meal_info_path
            )

            center_df = pd.read_csv(
                self.transformation_config.center_info_path
            )

            logger.info(
                "All datasets loaded successfully."
            )

            # --------------------------------------------------
            # 2. Merge and feature engineering
            # --------------------------------------------------

            train_df = self._prepare_data(
                train_df,
                meal_df,
                center_df
            )

            test_df = self._prepare_data(
                test_df,
                meal_df,
                center_df
            )

            logger.info(
                "Feature engineering completed."
            )

            # --------------------------------------------------
            # 3. Separate target
            # --------------------------------------------------

            target_column = "num_orders"

            X_train = train_df.drop(
                columns=[target_column]
            )

            y_train = train_df[target_column]

            X_test = test_df.drop(
                columns=[target_column]
            )

            y_test = test_df[target_column]

            # --------------------------------------------------
            # 4. Drop identifier columns
            # --------------------------------------------------

            drop_columns = [
                "id",
                "center_id",
                "meal_id"
            ]

            X_train = X_train.drop(
                columns=drop_columns
            )

            X_test = X_test.drop(
                columns=drop_columns
            )

            # --------------------------------------------------
            # 5. Define feature groups
            # --------------------------------------------------

            categorical_columns = [
                "category",
                "cuisine",
                "center_type"
            ]

            logger.info(
                f"Categorical columns: "
                f"{categorical_columns}"
            )

            # --------------------------------------------------
            # 6. Categorical preprocessing
            # --------------------------------------------------

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            # --------------------------------------------------
            # 7. Column Transformer
            # --------------------------------------------------

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_columns
                    )
                ],
                remainder="passthrough",
                sparse_threshold=1.0
            )

            # --------------------------------------------------
            # 8. Fit ONLY on training data
            # --------------------------------------------------

            logger.info(
                "Fitting preprocessor on training data."
            )

            X_train_transformed = (
                preprocessor.fit_transform(X_train)
            )

            # --------------------------------------------------
            # 9. Transform test data
            # --------------------------------------------------

            logger.info(
                "Transforming test data."
            )

            X_test_transformed = (
                preprocessor.transform(X_test)
            )

            # --------------------------------------------------
            # 10. Verify sparse output
            # --------------------------------------------------

            if not issparse(X_train_transformed):

                raise TypeError(
                    "Training transformation did not "
                    "produce a sparse matrix."
                )

            if not issparse(X_test_transformed):

                raise TypeError(
                    "Test transformation did not "
                    "produce a sparse matrix."
                )

            logger.info(
                f"Transformed training shape: "
                f"{X_train_transformed.shape}"
            )

            logger.info(
                f"Transformed testing shape: "
                f"{X_test_transformed.shape}"
            )

            # --------------------------------------------------
            # 11. Create artifact directory
            # --------------------------------------------------

            os.makedirs(
                self.transformation_config.root_dir,
                exist_ok=True
            )

            # --------------------------------------------------
            # 12. Save transformed features
            # --------------------------------------------------

            train_features_path = os.path.join(
                self.transformation_config.root_dir,
                "X_train.npz"
            )

            test_features_path = os.path.join(
                self.transformation_config.root_dir,
                "X_test.npz"
            )

            save_npz(
                train_features_path,
                X_train_transformed
            )

            save_npz(
                test_features_path,
                X_test_transformed
            )

            # --------------------------------------------------
            # 13. Save target values
            # --------------------------------------------------

            train_target_path = os.path.join(
                self.transformation_config.root_dir,
                "y_train.npy"
            )

            test_target_path = os.path.join(
                self.transformation_config.root_dir,
                "y_test.npy"
            )

            np.save(
                train_target_path,
                y_train.to_numpy()
            )

            np.save(
                test_target_path,
                y_test.to_numpy()
            )

            # --------------------------------------------------
            # 14. Save preprocessor
            # --------------------------------------------------

            joblib.dump(
                preprocessor,
                self.transformation_config.preprocessor_path
            )

            logger.info(
                "Preprocessor saved successfully."
            )

            logger.info(
                "Data Transformation completed successfully."
            )

            # --------------------------------------------------
            # 15. Return artifact paths
            # --------------------------------------------------

            return (
                train_features_path,
                test_features_path,
                train_target_path,
                test_target_path,
                self.transformation_config.preprocessor_path
            )

        except Exception as e:

            logger.exception(
                "Error occurred during Data Transformation."
            )

            raise CustomException(e, sys)


if __name__ == "__main__":

    transformation = DataTransformation()

    result = transformation.initiate_data_transformation(
        "artifacts/data_ingestion/train.csv",
        "artifacts/data_ingestion/test.csv"
    )

    print("\nTransformation Outputs:")

    for path in result:
        print(path)