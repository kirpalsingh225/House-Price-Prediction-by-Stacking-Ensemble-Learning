import pandas as pd
import pickle
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


class PredictPipeline:
    def __init__(self, n_citi, bed, bath, sqft, img_path):
        self.n_citi = n_citi
        self.bath = bath
        self.sqft = sqft
        self.bed = bed
        self.img_path = img_path

    def get_imgfeatures(self):
        autoencoder = load_model('saved_model/bottleneck.h5')
        image = cv.imread(self.img_path)
        image = cv.resize(image, (64,64))
        image = image/255.
        features = autoencoder.predict(image.reshape(1, image.shape[0], image.shape[0], 3))
        df = pd.DataFrame(data={"n_citi":[self.n_citi/414], "bed":[self.bed/12], "bath":[self.bath/36], "sqft":[self.sqft/11183]})
        summed_features = features[0][0]+features[0][1]+features[0][2]+features[0][3]+features[0][4]
        df["img_features"] = summed_features

        rf_model = pickle.load(open(r'saved_model\RandomForestModel.pkl', 'rb'))
        ada_model = pickle.load(open(r'saved_model\AdaBoostModel.pkl', 'rb'))
        extree_model = pickle.load(open(r'saved_model\ExtraTreeModel.pkl', 'rb'))
        lin_reg = pickle.load(open(r'saved_model\LinearRegressionModel.pkl', 'rb'))

        preds = []
        preds.append(rf_model.predict(df)[0])
        preds.append(ada_model.predict(df)[0])
        preds.append(extree_model.predict(df)[0])

        final_result = lin_reg.predict([preds])
        print(f"The price of the house is {final_result[0]*1999999}")


if __name__=="__main__":
    pp = PredictPipeline(48, 3, 2, 713, 'assets/image.png')
    pp.get_imgfeatures()