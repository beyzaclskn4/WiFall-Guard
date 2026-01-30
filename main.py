import numpy as np
from src.preprocess import load_csi_data, butter_lowpass_filter
from src.features import create_feature_matrix
from src.model import prepare_data, train_model, evaluate_model

# Dosya yolları ve parametreler
DATA_PATH = "data/annotations.csv"
WINDOW_SIZE = 100
