from variables import *
import numpy as np
import pandas as pd
import re
import pickle
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    df_cols = df.columns.values  

    Xinput = df[df_cols[1:]].values  
    Yinput = df[df_cols[0]].values  

    N, D = Xinput.shape 
    H = int(D ** 0.5)
    W = int(D ** 0.5)
    Xinput = Xinput.reshape(-1,H,W)/255.0

    Xinput = np.dstack([Xinput] * in_channels)
    Xinput = Xinput.reshape(N, H, W, in_channels)
    Xinput = np.pad(Xinput, padding, 'constant')

    Xinput, Yinput = shuffle(Xinput, Yinput)

    return Xinput, Yinput

def get_data():
    Xtrain, Ytrain = preprocess_data(train_path)
    Xtest , Ytest   = preprocess_data(test_path)
    return Xtrain, Ytrain, Xtest , Ytest