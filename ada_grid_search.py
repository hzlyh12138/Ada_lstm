# In[]
import os
import pickle
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.losses import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# In[]

# 读取数据集
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

lookback=8

# 归一化数据
scaler = MinMaxScaler(feature_range=(0, 1))
selected_data = scaler.fit_transform(selected_data)

# 将数据转换为适用于LSTM的输入格式
def create_dataset(dataset):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:i+lookback])
        y.append(dataset[i+lookback])
    return np.array(X), np.array(y).reshape(-1, 1)

X, y = create_dataset(selected_data)
# In[]
def custom_grid_search(create_model, param_grid, scoring, cv, record_file):
    best_score = None
    best_params = None
    best_model = None
    grid_scores = []
    with open(record_file, 'w') as file:
        file.write("record for n_estimators search \n")

    param_combinations = list(ParameterGrid(param_grid))
    total_combinations = len(param_combinations)
    lstm_model = KerasRegressor(build_fn=create_model(), epochs=50, batch_size=60, verbose=0)
    
    for i, params in enumerate(param_combinations):
        print(f"Grid Search Progress: {i+1}/{total_combinations}")

        scores = []
        
        for train_indices, val_indices in cv.split(X):
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            adaboost_lstm = AdaBoostRegressor(base_estimator=lstm_model, n_estimators=params['n_estimators'])
            adaboost_lstm.fit(X_train, y_train)

            # 进行预测
            y_pred = adaboost_lstm.predict(X_val)

            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_val = scaler.inverse_transform(y_val.reshape(-1, 1))
            score = scoring(y_val, y_pred)
            scores.append(score)
            print('score', np.mean(score))
            with open(record_file, "a") as f:
                f.write(f"kf_Score: {np.mean(score)}\n")
        pickle.dump(adaboost_lstm, open(f"adasaved_model/{params['n_estimators']}_model.sav", 'wb'))# 保存模型

        mean_score = np.mean(tf.concat(scores, axis=0))

        grid_scores.append((params, mean_score))
        
        if best_score is None or mean_score < best_score:
            best_score = mean_score
            best_params = params
            best_model = adaboost_lstm
            pickle.dump(best_model, open(f"adasaved_model/{params['n_estimators']}_best_model.sav", 'wb'))# 保存模型

        with open(record_file, "a") as f:
            f.write(f"Parameters: {params}, Score: {mean_score}\n\n")
    
    return best_params, best_model, grid_scores, best_score

# 定义Adaboost-LSTM模型
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(lookback, 1)))
    model.add(Dropout(0.2)) 
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

param_grid = {
    'n_estimators':[1,2,3,4,5,6,7,8,9] 
}

k = 5
kf = KFold(n_splits=k, shuffle=True)

best_params, best_model, grid_scores, best_score = custom_grid_search(create_model=create_lstm_model,
                                                          param_grid=param_grid,
                                                          scoring=mean_squared_error,
                                                          cv=kf,
                                                          record_file="record_grid_ada-lstm.txt")
# 输出最佳参数和对应的得分
print("Best Parameters:", best_params)
print("Best MSE Score:", best_score)

with open("record_grid_ada-lstm.txt", "a") as f:
    f.write(f"Best Parameters: {best_params}\n")
    f.write(f"Best MSE Score: {best_score}\n")