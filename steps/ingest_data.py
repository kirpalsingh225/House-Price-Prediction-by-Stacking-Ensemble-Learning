import logging
import pandas as pd
import numpy as np
from zenml import step

class IngestData:
    def __init__(self, data_path:str):
        self.data_path = data_path
    
    def get_data(self):
        logging.info(f"getting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path:str) -> pd.DataFrame:
    """
    Ingesting the data from data_path

    Args:
        data_path : path to the data
    Return:
        pd.DataFrame : the ingested data
    """