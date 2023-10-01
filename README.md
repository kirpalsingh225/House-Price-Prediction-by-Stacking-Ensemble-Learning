# House Price Prediction by Stacking Ensemble Learning
 The goal of this project is to develop a robust and highly accurate predictive model that can assist potential homebuyers, real estate professionals, and investors in estimating property values more effectively.

## Dataset

The dataset consists of the following features:-

    1. n_citi: Label Encoded citi values.

    2. Bed: The number of bedrooms.

    3. Bath: The number of Bathroom.

    4. sqft: The area of the house in square feet.

    5. img_features: The encoded image features of house images.

    6. Price: The price of the house.

## Machine Learning

- RandomForest Regressor - The data has been trained on random forest regressor and the model was able to achieve MAPE of 0.30 and mean squared error of 0.095.

- AdaBoost Regressor - AdaBoost Regressor has been able to achieve MAPE of 0.36 and mean squared error of 0.154.

- Extra Trees Regressor - ExtraTree Regressor has been able to achieve MAPE of 0.39 and mean squared error of 0.111.

- Linear Regression - Linear Regression has been trained on the predictions of the RandomForest, AdaBoost and ExtraTree Regressor to fine tune the predictions and able to achieve MAPE of 0.278 and mean squared error of 0.096.

## Run Locally

Clone the project

```bash
  git clone https://github.com/kirpalsingh225/House-Price-Prediction-by-Stacking-Ensemble-Learning
```

Go to the project directory

```bash
  cd House-Price-Prediction-by-Stacking-Ensemble-Learning
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python src/components/data_ingestion.py
```

## Architecture
![Architecture](https://github.com/kirpalsingh225/House-Price-Prediction-by-Stacking-Ensemble-Learning/assets/122300834/d0b32611-68c7-4830-abfc-1519dab41eee)

