import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras import metrics
from keras.callbacks import ModelCheckpoint

from imblearn.over_sampling import SMOTE

from pathlib import Path

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import math
import time

from sklearn.metrics import classification_report

from  scipy import ndimage, fft
from sklearn.preprocessing import normalize

from preprocess_data import LightFluxProcessor

np.random.seed(1)

def make_network(input_dim, output_dim, lrate):
    model = Sequential()
    model.add(Dense(units=1, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(units=output_dim, input_dim=1))
    model.add(Activation('sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=lrate),
                  metrics=['accuracy'])
    return model

def shuffle_data(x, y):
    data = shuffle(pd.DataFrame(x).join(pd.DataFrame(y))) # shuffle the points
    Xarray = np.array(data.drop(['LABEL'], axis=1))
    Yarray = np.array(data['LABEL']).reshape((len(data['LABEL']),1)) == 2
    return Xarray, Yarray

def main():
    train_df = pd.read_csv("../datasets/exoTrain.csv", encoding = "ISO-8859-1")
    test_df = pd.read_csv("../datasets/exoTest.csv", encoding = "ISO-8859-1")

    print(train_df)

    # Generate X and Y dataframe sets
    train_x_df = train_df.drop('LABEL', axis=1)
    train_y_df = train_df.LABEL
    test_x_df = test_df.drop('LABEL', axis=1)
    test_y_df = test_df.LABEL

    # Process dataset
    LFP = LightFluxProcessor(
        fourier=True,
        normalize=True,
        gaussian=True,
        standardize=True)
    train_x_df, test_x_df = LFP.process(train_x_df, test_x_df)

    # Load X and Y numpy arrays
    X_train, Y_train = shuffle_data(train_x_df, train_y_df)
    X_test, Y_test = shuffle_data(test_x_df, test_y_df)

    # Build model
    model = make_network(X_train.shape[1], Y_train.shape[1], 0.001)

    sampler = SMOTE(sampling_strategy = 1.0)
    X_train_sm, Y_train_sm = sampler.fit_sample(X_train, Y_train)
    #X_train_sm = X_train
    #Y_train_sm = Y_train

    print("Training...")
    history = model.fit(X_train_sm, Y_train_sm, epochs=50, batch_size=32)

    # Metrics
    train_outputs = model.predict(X_train, batch_size=32)
    test_outputs = model.predict(X_test, batch_size=32)

    #print(train_outputs)
    #print(test_outputs)

    train_outputs = np.rint(train_outputs)
    test_outputs = np.rint(test_outputs)

    accuracy_train = accuracy_score(Y_train, train_outputs)
    accuracy_test = accuracy_score(Y_test, test_outputs)

    precision_train = precision_score(Y_train, train_outputs)
    precision_test = precision_score(Y_test, test_outputs)

    recall_train = recall_score(Y_train, train_outputs)
    recall_test = recall_score(Y_test, test_outputs)

    print("train set error", 1.0 - accuracy_train)
    print("dev set error", 1.0 - accuracy_test)
    print("------------")
    print("precision_train", precision_train)
    print("precision_dev", precision_test)
    print("------------")
    print("recall_train", recall_train)
    print("recall_dev", recall_test)
    print("------------")
    print("------------")
    print("Train Set Positive Predictions", np.count_nonzero(train_outputs))
    print("Dev Set Positive Predictions", np.count_nonzero(test_outputs))
    #  Predicting 0's will give you error:
    print("------------")
    print("All 0's error train set", 37/5087)
    print("All 0's error dev set", 5/570)

main()
