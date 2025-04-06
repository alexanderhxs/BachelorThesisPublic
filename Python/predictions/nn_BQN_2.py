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
from keras import backend as K
from multiprocessing import Pool
import json
import scipy.stats as sps

# Accepts arguments:
#     cty (currently only DE), default: DE
#     distribution (Normal, StudentT, JSU, SinhArcsinh and NormalInverseGaussian), default: Normal

print('\n')
print(sys.executable)

trial = 4
d_degree = 12
INP_SIZE = 221


with open(f'/home/ahaas/BachelorThesis/params_trial_BQN_{trial}.json', 'r') as j:
    params = json.load(j)


if not os.path.exists(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_{trial}'):
    os.mkdir(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_{trial}')
print(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_{trial}')

# read data file
try:
    data = pd.read_csv('../../Datasets/DE.csv', index_col=0)
except:
    data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)

data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]

q_level_loss = np.arange(0.01, 1, 0.01)
B = np.zeros((d_degree+1, 99))
for d in range(d_degree+1):
    B[d, :] = sps.binom.pmf(d, d_degree, q_level_loss)

def qt_loss(y_true, y_pred):
    # Quantiles calculated via basis and increments
    B_tensor = K.constant(B, shape=(d_degree+1, 99), dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    y_pred = tf.reshape(y_pred, (-1, y_true.shape[1], d_degree+1))

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
    num_predictions, num_coeffs = alpha.shape
    alpha_reshaped = alpha.reshape(num_predictions, 24, d_degree + 1)

    # Initialize array to store results
    bernstein_quants = np.zeros((num_predictions, 24, 99))

    # Compute cumulative sum along axis=2 (coefficients)
    for i in range(num_predictions):
        cumsum_alpha = np.cumsum(alpha_reshaped[i], axis=1)
        bernstein_quants[i] = np.dot(cumsum_alpha, B)

    return bernstein_quants

def runoneday(inp):
    params, dayno = inp
    fc_period = int(1)
    df = data.iloc[dayno*24:dayno*24+1456*24+24*fc_period]

    if os.path.exists(os.path.join(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_{trial}', datetime.strftime(df.index[-24], '%Y-%m-%d'))):
        return
    # prepare the input/output dataframes
    Y = np.zeros((1456, 24))
    # Yf = np.zeros((1, 24)) # no Yf for rolling prediction
    for d in range(1456):
        Y[d, :] = df.loc[df.index[d*24:(d+1)*24], 'Price'].to_numpy()
    Y = Y[7:, :] # skip first 7 days
    # for d in range(1):
    #     Yf[d, :] = df.loc[df.index[(d+1092)*24:(d+1093)*24], 'Price'].to_numpy()
    X = np.zeros((1456+fc_period, INP_SIZE))
    for d in range(7, 1456+fc_period):
        X[d, :24] = df.loc[df.index[(d-1)*24:(d)*24], 'Price'].to_numpy() # D-1 price
        X[d, 24:48] = df.loc[df.index[(d-2)*24:(d-1)*24], 'Price'].to_numpy() # D-2 price
        X[d, 48:72] = df.loc[df.index[(d-3)*24:(d-2)*24], 'Price'].to_numpy() # D-3 price
        X[d, 72:96] = df.loc[df.index[(d-7)*24:(d-6)*24], 'Price'].to_numpy() # D-7 price
        X[d, 96:120] = df.loc[df.index[(d)*24:(d+1)*24], df.columns[1]].to_numpy() # D load forecast
        X[d, 120:144] = df.loc[df.index[(d-1)*24:(d)*24], df.columns[1]].to_numpy() # D-1 load forecast
        X[d, 144:168] = df.loc[df.index[(d-7)*24:(d-6)*24], df.columns[1]].to_numpy() # D-7 load forecast
        X[d, 168:192] = df.loc[df.index[(d)*24:(d+1)*24], df.columns[2]].to_numpy() # D RES sum forecast
        X[d, 192:216] = df.loc[df.index[(d-1)*24:(d)*24], df.columns[2]].to_numpy() # D-1 RES sum forecast
        X[d, 216] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[3]].to_numpy() # D-2 EUA
        X[d, 217] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[4]].to_numpy() # D-2 API2_Coal
        X[d, 218] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[5]].to_numpy() # D-2 TTF_Gas
        X[d, 219] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[6]].to_numpy() # D-2 Brent oil
        X[d, 220] = data.index[d].weekday()
    # '''
    # input feature selection
    colmask = [False] * 221
    if params['price_D-1']:
        colmask[:24] = [True] * 24
    if params['price_D-2']:
        colmask[24:48] = [True] * 24
    if params['price_D-3']:
        colmask[48:72] = [True] * 24
    if params['price_D-7']:
        colmask[72:96] = [True] * 24
    if params['load_D']:
        colmask[96:120] = [True] * 24
    if params['load_D-1']:
        colmask[120:144] = [True] * 24
    if params['load_D-7']:
        colmask[144:168] = [True] * 24
    if params['RES_D']:
        colmask[168:192] = [True] * 24
    if params['RES_D-1']:
        colmask[192:216] = [True] * 24
    if params['EUA']:
        colmask[216] = True
    if params['Coal']:
        colmask[217] = True
    if params['Gas']:
        colmask[218] = True
    if params['Oil']:
        colmask[219] = True
    if params['Dummy']:
        colmask[220] = True
    X = X[:, colmask]
    # '''
    Xf = X[-fc_period:, :]
    X = X[7:-fc_period, :]

    inputs = keras.Input(X.shape[1])
    last_layer = keras.layers.BatchNormalization()(inputs)

    # dropout
    dropout = params['dropout'] # trial.suggest_categorical('dropout', [True, False])
    if dropout:
        rate = params['dropout_rate'] # trial.suggest_float('dropout_rate', 0, 1)
        drop = keras.layers.Dropout(rate)(last_layer)
        last_layer = drop

    # regularization of 1st hidden layer,
    regularize_h1_activation = params['regularize_h1_activation']
    regularize_h1_kernel = params['regularize_h1_kernel']
    h1_activation_rate = (0.0 if not regularize_h1_activation 
                          else params['h1_activation_rate_l1'])
    h1_kernel_rate = (0.0 if not regularize_h1_kernel 
                      else params['h1_kernel_rate_l1'])
    # define 1st hidden layer with regularization
    hidden = keras.layers.Dense(params['neurons_1'], 
                                activation=params['activation_1'],
                                # kernel_initializer='ones',
                                kernel_regularizer=keras.regularizers.L1(h1_kernel_rate),
                                activity_regularizer=keras.regularizers.L1(h1_activation_rate))(last_layer)
    # regularization of 2nd hidden layer, 
    #activation - output, kernel - weights/parameters of input
    regularize_h2_activation = params['regularize_h2_activation']
    regularize_h2_kernel = params['regularize_h2_kernel']
    h2_activation_rate = (0.0 if not regularize_h2_activation 
                          else params['h2_activation_rate_l1'])
    h2_kernel_rate = (0.0 if not regularize_h2_kernel 
                      else params['h2_kernel_rate_l1'])
    # define 2nd hidden layer with regularization
    hidden = keras.layers.Dense(params['neurons_2'], 
                                activation=params['activation_2'],
                                # kernel_initializer='ones',
                                kernel_regularizer=keras.regularizers.L1(h2_kernel_rate),
                                activity_regularizer=keras.regularizers.L1(h2_activation_rate))(hidden)

    outputs = keras.layers.Dense(24*(d_degree+1), activation='softplus')(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']),
                  loss=qt_loss,
                  metrics=[qt_loss])

    # define callbacks
    callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
    perm = np.random.permutation(np.arange(X.shape[0]))
    VAL_DATA = .2
    trainsubset = perm[:int((1 - VAL_DATA)*len(perm))]
    valsubset = perm[int((1 - VAL_DATA)*len(perm)):]
    hist = model.fit(X[trainsubset], Y[trainsubset], epochs=1500, validation_data=(X[valsubset], Y[valsubset]), callbacks=callbacks, batch_size=32, verbose=False)


    predDF = pd.DataFrame(index=df.index[-24*fc_period:])
    predDF['alphas'] = pd.NA
    pred = model.predict(Xf)

    alpha_reshaped = pred.reshape(fc_period, 24, d_degree + 1)
    for day in range(fc_period):
        cumsum_alpha = np.cumsum(alpha_reshaped[day], axis=1)
        predDF.loc[predDF.index[day*24:(day+1)*24], 'alphas'] = [hour for hour in cumsum_alpha]
    predDF.to_csv(os.path.join(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_{trial}', datetime.strftime(df.index[-24], '%Y-%m-%d')))
    print(datetime.strftime(df.index[-24], '%Y-%m-%d'))
    print(predDF['alphas'].apply(lambda x: np.median(x)))
    print(np.mean(hist.history['loss']))
    print(np.mean(hist.history['val_loss']))
    return predDF


inputlist = [(params, day) for day in range(182, len(data) // 24 - 1456)]


start_time = datetime.now()
print(f'Program started at {start_time.strftime("%Y-%m-%d %H:%M:%S")}')

with Pool(4) as p:
    _ = p.map(runoneday, inputlist)

end_time = datetime.now()
compute_time = (end_time - start_time).total_seconds()
print(f'Program ended at {end_time.strftime("%Y-%m-%d %H:%M:%S")} \n computation time: {str(timedelta(seconds=compute_time))}')