import os 
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error


class ModelTrainer:
    def initiate_model_trainer(self, traindf, testdf, train_pfactor, test_pfactor):
        try:
            logging.info("creating models")
            rf_reg = RandomForestRegressor()
            ada_reg = AdaBoostRegressor()
            extree_reg = ExtraTreesRegressor()
            lin_reg = LinearRegression()

            logging.info("seprating independet and dependent feature")
            x_train = traindf.drop(["price"], axis=1)
            x_test = testdf.drop(["price"], axis=1)
            y_train = traindf["price"]
            y_test = testdf["price"]

            logging.info("fitting the ensemble models")
            rf_reg.fit(x_train, y_train)
            ada_reg.fit(x_train, y_train)
            extree_reg.fit(x_train, y_train)

            logging.info("predicting the values")
            rf_predict = rf_reg.predict(x_test)
            ada_predict = ada_reg.predict(x_test)
            extree_predict = extree_reg.predict(x_test)

            logging.info("building the linear regression model")
            predictions = pd.DataFrame(data={"p1":rf_predict, "p2":ada_predict, "p3":extree_predict, "target":y_test})

            lin_reg_x = predictions.drop(["target"],axis=1)
            lin_reg_y = predictions["target"]
            lin_reg.fit(lin_reg_x, lin_reg_y)

            logging.info("predicting the output")

            test_stacked_ensemble_predictions = lin_reg_x
            test_predictions = lin_reg.predict(test_stacked_ensemble_predictions)
            mse = mean_absolute_error(test_predictions, y_test)

            print(f"Stacked mean_squared_error {mse}")
            print(f"Mape stacked model {mean_absolute_percentage_error(test_predictions,y_test)}")
            print("------------------------------")
            print(f"Random Forest mean_squared_error {mean_absolute_error(rf_predict,y_test)}")
            print(f"Mape Random Forest model {mean_absolute_percentage_error(rf_predict,y_test)}")
            print("------------------------------")
            print(f"Adaboost mean_squared_error {mean_absolute_error(ada_predict,y_test)}")
            print(f"Mape Adaboost model {mean_absolute_percentage_error(ada_predict,y_test)}")
            print("------------------------------")
            print(f"Extra Tree mean_squared_error {mean_absolute_error(extree_predict,y_test)}")
            print(f"Mape Extra Tree model {mean_absolute_percentage_error(extree_predict,y_test)}")

            pickle.dump(rf_reg, open('saved_model/RandomForestModel.pkl', 'wb'))
            pickle.dump(ada_reg, open('saved_model/AdaBoostModel.pkl', 'wb'))
            pickle.dump(extree_reg, open('saved_model/ExtraTreeModel.pkl', 'wb'))
            pickle.dump(lin_reg, open('saved_model/LinearRegressionModel.pkl', 'wb'))

        except Exception as e:
            raise CustomException(e, sys)