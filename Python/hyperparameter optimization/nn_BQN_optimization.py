import pandas as pd
import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from datetime import datetime, timedelta
import tensorflow.compat.v2.keras as keras
import logging
import sys
import os
import optuna
import time
import sqlite3
from keras import backend as K
import scipy.stats as sps
# Accepts arguments:
#     cty (currently only DE), default: DE
#     distribution (Normal, StudentT, JSU, SinhArcsinh and NormalInverseGaussian), default: Normal

val_multi = 13 # int for # of re-trains - 1 corresponds to old approach
val_window = 364 // val_multi

INP_SIZE = 221
activations = ['sigmoid', 'relu', 'elu', 'tanh', 'softplus', 'softmax']
d_degree = 12

binopt = [True, False]

cty = 'DE'
storeDBintmp = False

if len(sys.argv) > 1:
    cty = sys.argv[1]
if len(sys.argv) > 2:
    distribution = sys.argv[2]
if len(sys.argv) > 3 and bool(sys.argv[3]):
    storeDBintmp = True

# read data file
data = pd.read_csv(f'/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]
data = data.iloc[:4*364*24] #take the first 4 years - 1456 days

q_level_loss = np.arange(0.01, 1, 0.01)
B = np.zeros((d_degree+1, 99))
for d in range(d_degree+1):
    B[d, :] = sps.binom.pmf(d, d_degree, q_level_loss)

def qt_loss(y_true, y_pred):
    # cast matrices to tensors and right data types
    B_tensor = K.constant(B, shape=(d_degree+1, 99), dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    #rehsape the predicted matrix
    y_pred = tf.reshape(y_pred, (-1, y_true.shape[1], d_degree+1))

    #calculate quantiles
    q = K.dot(K.cumsum(y_pred, axis=2), B_tensor)
    y_true = tf.expand_dims(y_true, axis=2)

    # Calculate CRPS
    err = tf.subtract(y_true, q)

    e1 = err * tf.constant(q_level_loss, shape=(1, 99), dtype=tf.float32)
    e2 = err * tf.constant(q_level_loss - 1, shape=(1, 99), dtype=tf.float32)

    scores = tf.maximum(e1, e2)
    scores = tf.reduce_mean(scores, axis=2)
    scores = tf.reduce_mean(scores, axis=1)
    return scores
def bern_quants(alpha):
    alpha = alpha.reshape(24, d_degree+1)
    return np.dot(np.cumsum(alpha, axis=1), B)
def objective(trial):
    # prepare the input/output dataframes
    Y = np.zeros((1456, 24))
    Yf = np.zeros((364, 24))
    for d in range(1456):
        Y[d, :] = data.loc[data.index[d*24:(d+1)*24], 'Price'].to_numpy()
    # Y = Y[7:, :] # skip first 7 days
    for d in range(364):
        Yf[d, :] = data.loc[data.index[(d+1092)*24:(d+1093)*24], 'Price'].to_numpy()
    #
    X = np.zeros((1092+364, INP_SIZE))
    for d in range(7, 1092+364):
        X[d, :24] = data.loc[data.index[(d-1)*24:(d)*24], 'Price'].to_numpy() # D-1 price
        X[d, 24:48] = data.loc[data.index[(d-2)*24:(d-1)*24], 'Price'].to_numpy() # D-2 price
        X[d, 48:72] = data.loc[data.index[(d-3)*24:(d-2)*24], 'Price'].to_numpy() # D-3 price
        X[d, 72:96] = data.loc[data.index[(d-7)*24:(d-6)*24], 'Price'].to_numpy() # D-7 price
        X[d, 96:120] = data.loc[data.index[(d)*24:(d+1)*24], data.columns[1]].to_numpy() # D load forecast
        X[d, 120:144] = data.loc[data.index[(d-1)*24:(d)*24], data.columns[1]].to_numpy() # D-1 load forecast
        X[d, 144:168] = data.loc[data.index[(d-7)*24:(d-6)*24], data.columns[1]].to_numpy() # D-7 load forecast
        X[d, 168:192] = data.loc[data.index[(d)*24:(d+1)*24], data.columns[2]].to_numpy() # D RES sum forecast
        X[d, 192:216] = data.loc[data.index[(d-1)*24:(d)*24], data.columns[2]].to_numpy() # D-1 RES sum forecast
        X[d, 216] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[3]].to_numpy() # D-2 EUA
        X[d, 217] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[4]].to_numpy() # D-2 API2_Coal
        X[d, 218] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[5]].to_numpy() # D-2 TTF_Gas
        X[d, 219] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[6]].to_numpy() # D-2 Brent oil
        X[d, 220] = data.index[d].weekday()
    # '''
    # input feature selection
    colmask = [False] * INP_SIZE
    if trial.suggest_categorical('price_D-1', [True]):
        colmask[:24] = [True] * 24
    if trial.suggest_categorical('price_D-2', binopt):
        colmask[24:48] = [True] * 24
    if trial.suggest_categorical('price_D-3', binopt):
        colmask[48:72] = [True] * 24
    if trial.suggest_categorical('price_D-7', binopt):
        colmask[72:96] = [True] * 24
    if trial.suggest_categorical('load_D', [True]):
        colmask[96:120] = [True] * 24
    if trial.suggest_categorical('load_D-1', binopt):
        colmask[120:144] = [True] * 24
    if trial.suggest_categorical('load_D-7', binopt):
        colmask[144:168] = [True] * 24
    if trial.suggest_categorical('RES_D', [True]):
        colmask[168:192] = [True] * 24
    if trial.suggest_categorical('RES_D-1', binopt):
        colmask[192:216] = [True] * 24
    if trial.suggest_categorical('EUA', binopt):
        colmask[216] = True
    if trial.suggest_categorical('Coal', binopt):
        colmask[217] = True
    if trial.suggest_categorical('Gas', binopt):
        colmask[218] = True
    if trial.suggest_categorical('Oil', binopt):
        colmask[219] = True
    if trial.suggest_categorical('Dummy', binopt):
        colmask[220] = True
    X = X[:, colmask]
    # '''
    Xwhole = X.copy()
    Ywhole = Y.copy()
    Yfwhole = Yf.copy()
    metrics_sub = []
    for train_no in range(val_multi):
        start = val_window * train_no
        X = Xwhole[start:1092+start, :]
        Xf = Xwhole[1092+start:1092+start+val_window, :]
        Y = Ywhole[start:1092+start, :]
        Yf = Ywhole[1092+start:1092+start+val_window, :]
        X = X[7:1092, :]
        Y = Y[7:1092, :]

        # begin building a model
        inputs = keras.Input(X.shape[1]) # <= INP_SIZE as some columns might have been turned off
        # batch normalization
        # we decided to always normalize the inputs
        batchnorm = True #trial.suggest_categorical('batch_normalization', [True, False])
        if batchnorm:
            norm = keras.layers.BatchNormalization()(inputs)
            last_layer = norm
        else:
            last_layer = inputs
        # dropout
        dropout = trial.suggest_categorical('dropout', binopt)
        if dropout:
            rate = trial.suggest_float('dropout_rate', 0, 1)
            drop = keras.layers.Dropout(rate)(last_layer)
            last_layer = drop
        # regularization of 1st hidden layer,
        #activation - output, kernel - weights/parameters of input
        regularize_h1_activation = trial.suggest_categorical('regularize_h1_activation', binopt)
        regularize_h1_kernel = trial.suggest_categorical('regularize_h1_kernel', binopt)
        h1_activation_rate = (0.0 if not regularize_h1_activation
                              else trial.suggest_float('h1_activation_rate_l1', 1e-5, 1e1, log=True))
        h1_kernel_rate = (0.0 if not regularize_h1_kernel
                          else trial.suggest_float('h1_kernel_rate_l1', 1e-5, 1e1, log=True))
        # define 1st hidden layer with regularization
        hidden = keras.layers.Dense(trial.suggest_int('neurons_1', 16, 1024, log=False),
                                    activation=trial.suggest_categorical('activation_1', activations),
                                    # kernel_initializer='ones',
                                    kernel_regularizer=keras.regularizers.L1(h1_kernel_rate),
                                    activity_regularizer=keras.regularizers.L1(h1_activation_rate))(last_layer)
        # regularization of 2nd hidden layer,
        #activation - output, kernel - weights/parameters of input
        regularize_h2_activation = trial.suggest_categorical('regularize_h2_activation', binopt)
        regularize_h2_kernel = trial.suggest_categorical('regularize_h2_kernel', binopt)
        h2_activation_rate = (0.0 if not regularize_h2_activation
                              else trial.suggest_float('h2_activation_rate_l1', 1e-5, 1e1, log=True))
        h2_kernel_rate = (0.0 if not regularize_h2_kernel
                          else trial.suggest_float('h2_kernel_rate_l1', 1e-5, 1e1, log=True))
        # define 2nd hidden layer with regularization
        hidden = keras.layers.Dense(trial.suggest_int('neurons_2', 16, 1024, log=False),
                                    activation=trial.suggest_categorical('activation_2', activations),
                                    # kernel_initializer='ones',
                                    kernel_regularizer=keras.regularizers.L1(h2_kernel_rate),
                                    activity_regularizer=keras.regularizers.L1(h2_activation_rate))(hidden)

        outputs = keras.layers.Dense(24 * (d_degree + 1), activation='softplus')(hidden)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)),
                      loss=qt_loss,
                      metrics=[qt_loss])
        # define callbacks
        callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
        model.fit(X, Y, epochs=1500, validation_data=(Xf, Yf), callbacks=callbacks, batch_size=32, verbose=0)

        metrics = model.evaluate(Xf, Yf)
        metrics_sub.append(metrics[0])
        # we optimize the returned value, -1 will always take the model with best MAE
    return np.mean(metrics_sub)

optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study_name = f'FINAL_DE_selection_BQN_4'
storage_directory = '/home/ahaas/BachelorThesis/hyperparameter'

os.makedirs(storage_directory, exist_ok=True)
db_file_path = os.path.join(storage_directory, f'{study_name}.db')

storage_name = f'sqlite:///{os.path.join(storage_directory, f"{study_name}.db")}'

if not os.path.isfile(db_file_path):
    conn = sqlite3.connect(db_file_path)
    conn.close()

study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
try:
    print(f'Trials so far: {len(study.trials)}')
except:
    print('Study not existing')
study.optimize(objective, n_trials=128, show_progress_bar=True, n_jobs=4)
best_params = study.best_params
print(best_params)
best_trial = study.best_trial