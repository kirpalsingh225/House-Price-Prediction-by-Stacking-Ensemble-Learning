import sys
import os
import random
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler


class DataTransformation:

    def data_transformation_object(self, train_path, test_path):
        '''
        This function is responsible for data transformation
        '''
        try:
            logging.info("reading the train and test data")
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

            logging.info("taking numerical features for dataset")
            train = train[["n_citi", "bed", "bath", "sqft", "price", "img_f1", "img_f2", "img_f3", "img_f4", "img_f5"]]
            test = test[["n_citi", "bed", "bath", "sqft", "price", "img_f1", "img_f2", "img_f3", "img_f4", "img_f5"]]

            logging.info("dropping img_f1, img_f3, img_f4 feature")
            train = train.drop(['img_f1', 'img_f3', 'img_f4'], axis=1)
            test = test.drop(['img_f1', 'img_f3', 'img_f4'], axis=1)

            # logging.info("removing outliers from target feature")
            # q1 = df["price"].quantile(0.25)
            # q3 = df["price"].quantile(0.75)
            # iqr = q3-q1
            # upper_limit = q3 + 1.5*iqr
            # lower_limit = q1 - 1.5*iqr
            # df = df[df["price"]<upper_limit].copy()

            logging.info("adding the image features to single feature")
            train["img_features"] = train["img_f2"] + train["img_f5"]
            train = train.drop(["img_f2", "img_f5"], axis=1)
            
            test["img_features"] = test["img_f2"] + test["img_f5"]
            test = test.drop(["img_f2", "img_f5"], axis=1)

            logging.info("normalising the data")
            citi_max_train = max(train["n_citi"])
            bed_max_train = max(train["bed"])
            bath_max_train = max(train["bath"])
            sqft_max_train = max(train["sqft"])
            price_max_train = max(train["price"])

            citi_max_test = max(test["n_citi"])
            bed_max_test = max(test["bed"])
            bath_max_test = max(test["bath"])
            sqft_max_test = max(test["sqft"])
            price_max_test = max(test["price"])

            train["n_citi"] = train["n_citi"]/citi_max_train
            train["bed"] = train["bed"]/bed_max_train
            train["bath"] = train["bath"]/bath_max_train
            train["sqft"] = train["sqft"]/sqft_max_train
            train["price"] = train["price"]/price_max_train

            test["n_citi"] = test["n_citi"]/citi_max_test
            test["bed"] = test["bed"]/bed_max_test
            test["bath"] = test["bath"]/bath_max_test
            test["sqft"] = test["sqft"]/sqft_max_test
            test["price"] = test["price"]/price_max_test
            print(price_max_test)

            return (train, test, price_max_train, price_max_test)

        except Exception as e:
            raise CustomException(e, sys)
        

    
        


        
