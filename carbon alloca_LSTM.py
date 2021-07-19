# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
# -

# ## Load Dataset 

data_path = r"Data/Data.xlsx"  # set file path

data = (
    pd.read_excel(
        data_path,
        sheet_name="DEA_data",
    )
    .set_index(["City name", "year", "region"])
    .loc[:, ["Population", "Fixed asset", "Energy consumption", "GDP", "CO2 emisson"]]
)

data.index.get_level_values('year').to_frame().describe()

# !tree .

pd.read_excel('Data/Data_lstm.xlsx')

# ## Data normalization 

# +
scaler = MinMaxScaler(feature_range=(0, 1))  # setup scaler

scaled = scaler.fit_transform(data)  # transform data into n*5 scaled arrays
scaled_df = pd.DataFrame(
    scaled, index=data.index, columns=data.columns
)  # set scaled arrays as dataframe
scaled_df.head()
# -

# ## Data pre-processing for LSTM

Input_lag = 16


def series_to_supervised(df, n_in=Input_lag, n_out=1, dropnan=True):

    n_vars = 1 if type(df) is list else df.shape[1]
    # if number of variables is 1 when input data type is a list, else the number is the shape of data

    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))  # shift dataframe values forward for i period
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# ## Train-test data split

def train_test_split(data, n_var=Input_lag * 5):
    """
    split train and test groups by 80% and 20%
    """
    size = int(len(data) * 0.8)
    # for train data will be collected from each country's data which index is from 0-size (80%)
    x_train = data.iloc[0:size, 0:n_var]
    # for test data will be collected from each country's  data which index is from size to the end (20%)
    x_test = data.iloc[size:, 0:n_var]
    y_train = data.iloc[0:size, n_var:]
    y_test = data.iloc[size:, n_var:]
    return x_train, x_test, y_train, y_test


def reshape(dataframe):
    """
    Reshape dataframe into np.array fitting to LSTM
    """
    array = dataframe.values.reshape(dataframe.shape[0], 1, dataframe.shape[1])
    return array


def data_process(province, scaled_df=scaled_df):
    data = scaled_df[
        scaled_df.index.get_level_values("City name") == province
    ]  # data transferred to (n_varX+n_varY)*(12-n_varX/5)
    data = series_to_supervised(
        data[data.index.get_level_values("City name") == province]
    )
    x_train, x_test, y_train, y_test = train_test_split(data)
    x_train_array, x_test_array, y_train_array, y_test_array = (
        reshape(x_train),
        reshape(x_test),
        reshape(y_train),
        reshape(y_test),
    )
    return x_train_array, x_test_array, y_train_array, y_test_array


# +
province_list = list(
    dict.fromkeys([i[0] for i in scaled_df.index])
)  # get the list of province names, drop duplicate names while keep order

X_train = []
X_test = []
Y_train = []
Y_test = []

for province in province_list:
    x_train_array, x_test_array, y_train_array, y_test_array = data_process(province)

    X_train.append(x_train_array)
    X_test.append(x_test_array)
    Y_train.append(y_train_array)
    Y_test.append(y_test_array)  # get train & test sets


# -

# ##  Model Setup

def mean_absolute_percentage_error(y_true, y_pred):

    ## Note: does not handle mix 1d representation
    # if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# 定义损失函数


def model_train(
    i,
    neurons_First,
    neurons_Second,
    layer=1,
    dropout=0.2,
    learning_rate=0.01,
    epochs=200,
    loss="mae",
):
    """
    Train LSTM model for different provinces to predict their population, GDP, capital stock, CO2, enery consumption
    """
    # design network for confirmed cases data
    model = Sequential()
    model.add(
        LSTM(
            neurons_First,
            activation="relu",
            input_shape=(
                X_train[i].shape[1],
                X_train[i].shape[2],
            ),
            return_sequences=(
                True if layer == 2 else False
            ),  # LSTM layer requires 3D input, by using 'return_sequeces' argument,
            # the layer returns LSTM output as same as input
            dropout=dropout,
        )
    )  # add LSTM layer

    if layer == 2:
        model.add(Dropout(dropout))

        model.add(LSTM(neurons_Second, activation="relu"))  # stacked LSTM model
        model.add(Dropout(dropout))

    model.add(Dense(5))  # add output layer

    optimizer = Adam(lr=learning_rate, decay=1e-6)
    model.compile(loss=loss, optimizer=optimizer)  # add loss function and optimizer

    # fit network
    model_train = model.fit(
        X_train[i],
        Y_train[i],
        epochs=epochs,
        validation_data=(X_test[i], Y_test[i]),
        verbose=0,
        shuffle=False,
    )

    # make a prediction
    yhat = model.predict(X_test[i])

    # invert scaling for forecast
    inv_yhat = scaler.inverse_transform(yhat)
    inv_y = scaler.inverse_transform(
        Y_test[i].reshape(Y_test[i].shape[0], Y_test[i].shape[2])
    )
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    mape = mean_absolute_percentage_error(inv_y, inv_yhat)

    # save model
    path = "model/SG_DEA"
    os.chdir(path)
    if (province_list[i] + "_lstm.h5") not in os.listdir(path):
        model.save(province_list[i] + "_lstm.h5")

    return rmse, mape, yhat


# +
# # %mkdir model/
# -

length = len(province_list)
dropout = 0
neurons_First = 40
neurons_Second = 20
layer = 1
lr = 0.01
epochs = 400
loss = "mae"

# +
mape_list, rmse_list = [], []

for province_i in range(length):
    rmse, mape, yhat = model_train(
        province_i,
        neurons_First,
        neurons_Second,
        layer,
        dropout=dropout,
        learning_rate=lr,
        epochs=epochs,
        loss=loss,
    )
    mape_list.append(mape)
    rmse_list.append(rmse)
    print("{}: rmse {:.3f} \nmape {:.3f}".format(province_list[province_i], rmse, mape))

sum(mape_list) / length, sum(rmse_list) / length
# -

# ## Model prediction

model_file_list = os.listdir(r"E:\desktop\PythonScipt\keras_models\ZSG_DEA")


def model_load(i):
    model_file = [
        file_name for file_name in model_file_list if province_list[i] in file_name
    ][0]
    model = load_model(model_file)
    yhat = model.predict(X_test[i])
    return model, yhat


def model_prediction(i, model, y_predict, scaled_df=scaled_df):

    # predict next timestep based on previous times steps
    raw_data = scaled_df[
        scaled_df.index.get_level_values("City name") == province_list[i]
    ]

    tuples = [(province_list[i], raw_data.index[-1][1] + 1, raw_data.index[-1][2])]
    index = pd.MultiIndex.from_tuples(tuples)

    y_predict_df = pd.DataFrame(
        y_predict, columns=raw_data.columns, index=index
    )  # transfer ndarray into dataframe

    data_update_df = pd.concat(
        [raw_data, y_predict_df]
    )  # concatnate original data with predicted data

    data_update = series_to_supervised(data_update_df)
    x_train, x_test, y_train, y_test = train_test_split(data_update)
    x_train_array, x_test_array, y_train_array, y_test_array = (
        reshape(x_train),
        reshape(x_test),
        reshape(y_train),
        reshape(y_test),
    )

    y_predict = model.predict(x_test_array)

    return y_predict, data_update_df


def future_prediction(i, future_period):

    y_list = []
    for period in range(future_period):
        if period == 0:
            model, yhat = model_load(i)
            y_predict, data_update_df = model_prediction(i, model, yhat)
            row, col = y_predict.shape[0], y_predict.shape[1]
            y_list.append(y_predict)
        else:
            y_predict, data_update_df = model_prediction(
                i, model, y_predict, data_update_df
            )
            y_predict = np.reshape(y_predict[-1], (row, col))
            y_list.append(y_predict)

    return data_update_df


def inverse_df(i, future_period):
    data_update_df = future_prediction(i, future_period)
    data_final = scaler.inverse_transform(data_update_df)
    data_final_df = pd.DataFrame(
        data_final,
        columns=data_update_df.columns,
        index=range(2000, 2000 + data_final.shape[0]),
    )
    return data_final_df


# +
data_dict = {}
for i in range(len(province_list)):
    data_dict[province_list[i]] = inverse_df(i, 13)

data_concat = pd.concat(data_dict)
# -

path = r"D:\tencent files\chrome Download\Research\DEA\DEA_carbon market\Data"
with open(os.path.join(path, "Data_lstm.pickle"), "wb") as data:
    pickle.dump(data_concat, data)

data_concat.to_excel(
    r"D:\tencent files\chrome Download\Research\DEA\DEA_carbon market\Data\Data_lstm.xlsx"
)
