from src.components.data_ingestion import DataTransformation
import pandas as pd
import pickle
from keras.models import load_model


class PredictPipeline:
    def __init__(self, n_citi, bath, sqft, img_path):
        self.n_citi = n_citi
        self.bath = bath
        self.sqft = sqft
        self.img_path = img_path

    def get_imgfeatures(self):
        autoencoder = load_model('saved_model\bottleneck.h5')
        print(autoencoder.summary())

if __name__=="__main__":
    pp = PredictPipeline()
    pp.get_imgfeatures()