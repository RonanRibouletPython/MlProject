import sys
import os

from dataclasses import dataclass

import numpy as np 
import pandas as pd

# import the save_object module from the utils file
from src.utils import save_object

# SkLearn
# Used for the column transformations
from sklearn.compose import ColumnTransformer
# Used for missing values
from sklearn.impute import SimpleImputer
# Used for the pipeline 
from sklearn.pipeline import Pipeline
# Preprocess the features and labels
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Custom exception handling 
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        
        # This function is responsible for data transformation
        
        try:
            # Create the distinction between the numerical and categorical features
            numerical_columns = [
                "writing_score", 
                "reading_score"
                ]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                ]

            # This pipeline is intended for the training data
            num_pipeline = Pipeline(
                steps=[
                    # Imputer is handling the missing values and replacing them with the median value                       
                    ("imputer", SimpleImputer(strategy = "median")),
                    # Scaler is removing the mean and scaling to unit variance
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    # Imputer is handling the missing value using the most frequent value along each column
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    # Encode categorical features as a one-hot numeric array
                    ("one_hot_encoder", OneHotEncoder()),
                    # Need to remove the mean scaler for cat columns
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical and numerical categories standard scaling and encoding completed")
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combination of the cat and num pipelines
            preprocessor = ColumnTransformer(
                    [
                        ("num_pipeline", num_pipeline, numerical_columns),
                        ("cat_pipelines", cat_pipeline, categorical_columns)
                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            # Create the pandas dataframes
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Obtained the preprocessing object")

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Drop the target column from the input feature dataframe
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            # Isolate the target 
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr  =preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
                ]

            logging.info("Saved preprocessing object.")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
        
        except Exception as e:
            raise CustomException(e, sys)