import pandas as pd
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
import matplotlib.pyplot as plt

from keras import Model
from keras.layers import Lambda, Input, Dropout, Flatten, LSTM, Concatenate, Bidirectional, Conv1D, MaxPooling1D
from keras import backend as K
from keras.callbacks import TensorBoard
from time import time
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from time import time


def conv_1d_model_generator(n_timesteps, n_features, n_outputs, provided_input=None):
    # Input
    if provided_input == None:
        x = Input((n_timesteps, n_features))
    else:
        x = provided_input

    # Conv
    cnn1_1 = Conv1D(filters=100, kernel_size=10, activation='relu')(x)
    cnn1_2 = Conv1D(filters=100, kernel_size=10, activation='relu')(cnn1_1)
    cnn1_3 = MaxPooling1D(3)(cnn1_2)
    cnn1_4 = Conv1D(filters=160, kernel_size=10, activation='relu')(cnn1_3)
    # cnn1_5 = GlobalAveragePooling1D()(cnn1_4)
    cnn1_5 = MaxPooling1D(3)(cnn1_4)
    cnn1_6 = Flatten()(cnn1_5)
    cnn1_7 = Dense(20, activation='relu')(cnn1_6)
    cnn1_8 = Dense(n_outputs, activation='softmax')(cnn1_7)

    return x, cnn1_6, cnn1_8


def dense_fully_connected_model_generator(n_timesteps, n_features, n_outputs, provided_input=None):
    # Input
    if provided_input == None:
        x = Input((n_timesteps, n_features))
    else:
        x = provided_input

    # Dense
    dense_1 = Dense(250, activation='relu')(x)
    dense_2 = Dropout(0.4)(dense_1)
    dense_3 = Dense(250, activation='relu')(dense_2)
    dense_4 = Dropout(0.2)(dense_3)
    dense_5 = Dense(30, activation='relu')(dense_4)
    dense_6 = Flatten()(dense_5)
    dense_7 = Dense(n_outputs, activation='softmax')(dense_6)

    return x, dense_7, dense_6

def dense_1d_model_generator(n_timesteps, n_features, n_outputs, provided_input=None):
    # Input
    if provided_input == None:
        x = Input((n_timesteps, n_features))
    else:
        x = provided_input

    # Dense
    dense_1 = Lambda(lambda x: K.tf.unstack(x, axis=2))(x)
    dense_2 = [Dense(20, activation='relu')(x) for x in dense_1]
    dense_3 = Lambda(lambda x: K.stack(x, axis=2))(dense_2)
    dense_4 = Dropout(0.1)(dense_3)
    dense_5 = Flatten()(dense_4)
    dense_6 = Dense(250, activation='relu')(dense_5)
    dense_7 = Dense(20, activation='relu')(dense_6)
    dense_8 = Dense(n_outputs, activation='softmax')(dense_7)

    return x, dense_8, dense_6


def lstm_model_generator(n_timesteps, n_features, n_outputs, provided_input=None):
    # Input
    if provided_input == None:
        x = Input((n_timesteps, n_features))
    else:
        x = provided_input

    # LSTM
    lstm_1 = LSTM(100, activation='relu', input_shape=(n_timesteps, n_features))(x)
    lstm_2 = Dropout(0.5)(lstm_1)
    lstm_3 = Dense(100, activation='relu')(lstm_2)
    lstm_4 = Dense(n_outputs, activation='softmax', name='lstm_out')(lstm_3)

    return x, lstm_4, lstm_3


def hybrid_ens_generator(n_timesteps, n_features, n_outputs, ):
    dense_in, dense_out, dense_int = dense_1d_model_generator(n_timesteps, n_features, n_outputs)
    lstm_in, lstm_out, lstm_int = lstm_model_generator(n_timesteps, n_features, n_outputs, provided_input=dense_in)

    ens_1 = Concatenate(axis=1)([lstm_int, dense_int])
    ens_2 = Dense(n_outputs, activation='softmax')(ens_1)

    return dense_in, ens_2, ens_1


def dense_model_2_generator(n_timesteps, n_features, n_outputs, provided_input=None):
    # Input
    if provided_input == None:
        x = Input((n_timesteps, n_features))
    else:
        x = provided_input

    # Dense
    dense_1 = Lambda(lambda x: K.tf.unstack(x, axis=2))(x)
    dense_2 = [Dense(30, activation='relu')(x) for x in dense_1]
    dense_3 = Lambda(lambda x: K.stack(x, axis=2))(dense_2)
    dense_4 = Dropout(0)(dense_3)
    dense_5 = Flatten()(dense_4)
    dense_6 = Dense(300, activation='relu')(dense_5)
    dense_7 = Dropout(0.4)(dense_6)
    dense_8 = Dense(300, activation='relu')(dense_7)
    dense_9 = Dense(20, activation='relu')(dense_8)
    dense_10 = Dense(n_outputs, activation='softmax', name='dense_2_out')(dense_9)

    return x, dense_10, dense_8


def lstm_model_2_generator(n_timesteps, n_features, n_outputs, provided_input=None):
    # Input
    if provided_input == None:
        x = Input((n_timesteps, n_features))
    else:
        x = provided_input

    # LSTM
    lstm_1 = Bidirectional(LSTM(100, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True))(x)
    lstm_2 = Bidirectional(LSTM(100, activation='relu'))(lstm_1)
    lstm_3 = Dropout(0.2)(lstm_2)
    lstm_4 = Dense(120, activation='relu')(lstm_3)
    lstm_5 = Dense(n_outputs, activation='softmax', name='lstm_out')(lstm_4)

    return x, lstm_5, lstm_4
