import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from sklearn.model_selection import KFold
import os
import warnings
from tools import calculate_nse, calculate_mse, calculate_rmse, calculate_mae, calculate_mape, calculate_si

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

datalabel210103 = pd.read_excel('fakoulabeldata/202101-03.xls')
datalabel210406 = pd.read_excel('fakoulabeldata/202104-06.xls')
datalabel210709 = pd.read_excel('fakoulabeldata/202107-09.xls')
datalabel220103 = pd.read_excel('fakoulabeldata/202201-03.xls')
datalabel220406 = pd.read_excel('fakoulabeldata/202204-06.xls')
datalabel220709 = pd.read_excel('fakoulabeldata/202207-09.xls')
datalabel221012 = pd.read_excel('fakoulabeldata/202210-12.xls')
datalabel230103 = pd.read_excel('fakoulabeldata/202301-03.xls')
datalabel230406 = pd.read_excel('fakoulabeldata/202304-06.xls')
datalabel230709 = pd.read_excel('fakoulabeldata/202307-09.xls')

frame = (datalabel210103.iloc[:,:65333],datalabel210406.iloc[:,:65529],datalabel210709.iloc[:,:65530],
       datalabel220103.iloc[:,:65531],datalabel220406.iloc[:,:65525],datalabel220709.iloc[:,:65517],
       datalabel221012.iloc[:,:65522],datalabel230103.iloc[:,:65503],datalabel230406.iloc[:,:65495],
       datalabel230709.iloc[:,:65534])

all_data = pd.concat(frame, ignore_index=True)
selected_data = all_data.iloc[:,1]
selected_data = np.r_[selected_data].reshape(-1, 1)


scaler = MinMaxScaler(feature_range=(0, 1))
selected_data = scaler.fit_transform(selected_data)

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:i+lookback])
        y.append(dataset[i+lookback])
    return np.array(X), np.array(y).reshape(-1, 1)

k = 5
kf = KFold(n_splits=k, shuffle=True)

lookback_range = [3, 4, 5, 6, 7, 8, 9, 10, 15]

best_score = float('inf')
best_lookback = None
def create_model(lookback):
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(lookback, 1)))
    model.add(Dropout(0.2)) 
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    
for lookback in lookback_range:
    X, Y = create_dataset(selected_data, lookback)
    X = np.reshape(X, (X.shape[0], -1))
    with open('record_lstm_lookback.txt', 'a') as file:
        file.write(f"Lookback: {lookback}\n")
    fold = 1
    mse_scores = []
    for train_index, test_index in kf.split(X):
        print(f"Fold: {fold}")
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model = create_model(lookback)
        model.fit(X_train, Y_train, epochs=10, batch_size=96, verbose=0)

        Y_pred = model.predict(X_test)
        Y_pred = scaler.inverse_transform(Y_pred.reshape(-1, 1))
        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))
        mse = mean_squared_error(Y_test, Y_pred)
        mse_scores.append(mse)
        print(f"MSE: {mse}")
        print("fold_MSE:", tf.reduce_mean(mse))
        fold += 1
        with open('record_lstm_lookback.txt', 'a') as file:
            file.write(f"fold_MSE: {tf.reduce_mean(mse)}\n")

    merged_tensor = tf.concat(mse_scores, axis=0)
    mean_mse = tf.reduce_mean(merged_tensor)
    print("Lookback:", lookback)
    print(f"Mean MSE: {mean_mse} \n")
    with open('record_lstm_lookback.txt', 'a') as file:
        file.write(f"Mean MSE: {mean_mse}\n\n")
    if mean_mse < best_score:
        best_lookback = lookback
        best_score = mean_mse

