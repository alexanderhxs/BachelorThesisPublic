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

# Accepts arguments:
#     cty (currently only DE), default: DE
#     distribution (Normal, StudentT, JSU, SinhArcsinh and NormalInverseGaussian), default: Normal

distribution = 'JSU'
paramcount = {'Normal': 2,
              'JSU': 4,
              'Point': None}
val_multi = 13 # int for # of re-trains - 1 corresponds to old approach
val_window = 364 // val_multi

if not os.path.exists(f'../../hyperparameter'):
    os.mkdir(f'../../hyperparameter')

INP_SIZE = 14
activations = ['sigmoid', 'relu', 'elu', 'tanh', 'softplus', 'softmax']

binopt = [True, False]

cty = 'DE'
storeDBintmp = False

if len(sys.argv) > 1:
    cty = sys.argv[1]
if len(sys.argv) > 2:
    distribution = sys.argv[2]
if len(sys.argv) > 3 and bool(sys.argv[3]):
    storeDBintmp = True

print(cty, distribution)

if cty != 'DE':
    raise ValueError('Incorrect country')
if distribution not in paramcount:
    raise ValueError('Incorrect distribution')

# read data file
data = pd.read_csv(f'/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]
data = data.iloc[:4*364*24] #take the first 4 years - 1456 days

def objective(trial):
    # prepare the input/output dataframes
    Y = data.iloc[:1456*24, 0].values
    # Y = Y[7:, :] # skip first 7 days
    Yf = data.iloc[1092*24:1456*24, :].values
    #
    X = np.zeros(((1092*24+364*24), INP_SIZE))
    for d in range(7*24, 1092*24+364*24):
        X[d, 0] = data.iloc[(d - 1 * 24), 0]  # D-1 price
        X[d, 1] = data.iloc[(d - 2 * 24), 0]  # D-2 price
        X[d, 2] = data.iloc[(d - 3 * 24), 0]  # D-3 price
        X[d, 3] = data.iloc[(d - 7 * 24), 0]  # D-7 price
        X[d, 4] = data.iloc[d, 1]  # D load forecast
        X[d, 5] = data.iloc[(d - 1 * 24), 1]  # D-1 load forecast
        X[d, 6] = data.iloc[(d - 7 * 24), 1]  # D-7 load forecast
        X[d, 7] = data.iloc[d, 2]  # D RES sum forecast
        X[d, 8] = data.iloc[(d - 1 * 24), 2]  # D-1 RES sum forecast
        X[d, 9] = data.iloc[(d - 2 * 24), 3]  # D-2 EUA
        X[d, 10] = data.iloc[(d - 2 * 24), 4]  # D-2 API2_Coal
        X[d, 11] = data.iloc[(d - 2 * 24), 5]  # D-2 TTF_Gas
        X[d, 12] = data.iloc[(d - 2 * 24), 6]  # D-2 Brent oil
        X[d, 13] = data.index[d].weekday()
    # '''
    # input feature selection
    colmask = [False] * INP_SIZE
    if trial.suggest_categorical('price_D-1', [True]):
        colmask[0] = True
    if trial.suggest_categorical('price_D-2', binopt):
        colmask[1] = True
    if trial.suggest_categorical('price_D-3', binopt):
        colmask[2] = True
    if trial.suggest_categorical('price_D-7', binopt):
        colmask[3] = True
    if trial.suggest_categorical('load_D', [True]):
        colmask[4] = True
    if trial.suggest_categorical('load_D-1', binopt):
        colmask[5] = True
    if trial.suggest_categorical('load_D-7', binopt):
        colmask[6] = True
    if trial.suggest_categorical('RES_D', [True]):
        colmask[7] = True
    if trial.suggest_categorical('RES_D-1', binopt):
        colmask[8] = True
    if trial.suggest_categorical('EUA', binopt):
        colmask[9] = True
    if trial.suggest_categorical('Coal', binopt):
        colmask[10] = True
    if trial.suggest_categorical('Gas', binopt):
        colmask[11] = True
    if trial.suggest_categorical('Oil', binopt):
        colmask[12] = True
    if trial.suggest_categorical('Dummy', binopt):
        colmask[13] = True

    X = X[:, colmask]
    # '''
    Xwhole = X.copy()
    Ywhole = Y.copy()
    Yfwhole = Yf.copy()
    metrics_sub = []

    #generate params
    params_suggested = {
        'dropout': trial.suggest_categorical('dropout', binopt),
        'dropout_rate': trial.suggest_float('dropout_rate', 0, 1),
        'regularize_h1_activation': trial.suggest_categorical('regularize_h1_activation', binopt),
        'regularize_h1_kernel': trial.suggest_categorical('regularize_h1_kernel', binopt),
        'h1_activation_rate_l1': trial.suggest_float('h1_activation_rate_l1', 1e-5, 1e1, log=True),
        'h1_kernel_rate_l1': trial.suggest_float('h1_kernel_rate_l1', 1e-5, 1e1, log=True),
        'neurons_1': trial.suggest_int('neurons_1', 16, 256, log=False),
        'activation_1': trial.suggest_categorical('activation_1', activations),
        'regularize_h2_activation': trial.suggest_categorical('regularize_h2_activation', binopt),
        'regularize_h2_kernel': trial.suggest_categorical('regularize_h2_kernel', binopt),
        'h2_activation_rate_l1': trial.suggest_float('h2_activation_rate_l1', 1e-5, 1e1, log=True),
        'h2_kernel_rate_l1': trial.suggest_float('h2_kernel_rate_l1', 1e-5, 1e1, log=True),
        'neurons_2': trial.suggest_int('neurons_2', 16, 256, log=False),
        'activation_2': trial.suggest_categorical('activation_2', activations),
        'regularize_loc': trial.suggest_categorical('regularize_loc', binopt),
        'regularize_scale': trial.suggest_categorical('regularize_scale' , binopt),
        'regularize_tailweight': trial.suggest_categorical('regularize_tailweight', binopt),
        'regularize_skewness': trial.suggest_categorical('regularize_skewness', binopt),
        'loc_rate_l1': trial.suggest_float('loc_rate_l1', 1e-5, 1e1, log=True),
        'scale_rate_l1': trial.suggest_float('scale_rate_l1', 1e-5, 1e1, log=True),
        'tailweight_rate_l1': trial.suggest_float('tailweight_rate_l1', 1e-5, 1e1, log=True),
        'skewness_rate_l1': trial.suggest_float( 'skewness_rate_l1', 1e-5, 1e1, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    }
    X = Xwhole[:1092*24, :]
    Xf = Xwhole[1092*24:, :]
    Y = Ywhole[:1092*24]
    Yf = Ywhole[1092*24:]
    X = X[7*24:1092*24, :]
    Y = Y[7*24:1092*24]

    for tm in range(24):
        X_tm = X[tm::24]
        Y_tm = Y[tm::24]
        Xf_tm = Xf[tm::24]
        Yf_tm = Yf[tm::24]
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
        dropout = params_suggested['dropout']
        if dropout:
            rate = params_suggested['dropout_rate']
            drop = keras.layers.Dropout(rate)(last_layer)
            last_layer = drop
        # regularization of 1st hidden layer,
        #activation - output, kernel - weights/parameters of input
        regularize_h1_activation = params_suggested['regularize_h1_activation']
        regularize_h1_kernel = params_suggested['regularize_h1_kernel']
        h1_activation_rate = (0.0 if not regularize_h1_activation
                              else params_suggested['h1_activation_rate_l1'])
        h1_kernel_rate = (0.0 if not regularize_h1_kernel
                          else params_suggested['h1_kernel_rate_l1'])
        # define 1st hidden layer with regularization
        hidden = keras.layers.Dense(params_suggested['neurons_1'],
                                    activation=params_suggested['activation_1'],
                                    # kernel_initializer='ones',
                                    kernel_regularizer=keras.regularizers.L1(h1_kernel_rate),
                                    activity_regularizer=keras.regularizers.L1(h1_activation_rate))(last_layer)
        # regularization of 2nd hidden layer,
        #activation - output, kernel - weights/parameters of input
        regularize_h2_activation = params_suggested['regularize_h2_activation']
        regularize_h2_kernel = params_suggested['regularize_h2_kernel']
        h2_activation_rate = (0.0 if not regularize_h2_activation
                              else params_suggested['h2_activation_rate_l1'])
        h2_kernel_rate = (0.0 if not regularize_h2_kernel
                          else params_suggested['h2_kernel_rate_l1'])
        # define 2nd hidden layer with regularization
        hidden = keras.layers.Dense(params_suggested['neurons_2'],
                                    activation=params_suggested['activation_2'],
                                    # kernel_initializer='ones',
                                    kernel_regularizer=keras.regularizers.L1(h2_kernel_rate),
                                    activity_regularizer=keras.regularizers.L1(h2_activation_rate))(hidden)


        # now define parameter layers with their regularization
        param_layers = []
        param_names = ["loc", "scale", "tailweight", "skewness"]
        for p in range(paramcount[distribution]):

            regularize_param_kernel = params_suggested['regularize_'+param_names[p]]
            param_kernel_rate = (0.0 if not regularize_param_kernel
                                 else params_suggested[param_names[p]+'_rate_l1'])
            param_layers.append(keras.layers.Dense(
                1, activation='linear',
                kernel_regularizer=keras.regularizers.L1(param_kernel_rate))(hidden))
        # concatenate the parameter layers to one
        linear = tf.keras.layers.concatenate(param_layers)
        # define outputs
        if distribution == 'Normal':
            outputs = tfp.layers.DistributionLambda(
                    lambda t: tfd.Normal(
                        loc=t[..., 0],
                        scale = 1e-3 + 3 * tf.math.softplus(t[..., 1])))(linear)
        elif distribution == 'JSU':
            outputs = tfp.layers.DistributionLambda(
                    lambda t: tfd.JohnsonSU(
                        loc=t[..., 0],
                        scale=1e-3 + 3 * tf.math.softplus(t[..., 1]),
                        tailweight= 1 + 3 * tf.math.softplus(t[..., 2]),
                        skewness=t[..., 3]))(linear)
        else:
            raise ValueError(f'Incorrect distribution {distribution}')
        model = keras.Model(inputs = inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(params_suggested['learning_rate']),
                      loss=lambda y, rv_y: -rv_y.log_prob(y),
                      metrics='mae')

        # define callbacks
        callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
        model.fit(X_tm, Y_tm, epochs=1500, validation_data=(Xf_tm, Yf_tm), callbacks=callbacks, batch_size=32, verbose=0)

        metrics = model.evaluate(Xf_tm, Yf_tm) # for point its a list of one [loss, MAE]
        metrics_sub.append(metrics[0])
        # we optimize the returned value, -1 will always take the model with best MAE
    return np.mean(metrics_sub)

optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study_name = f'FINAL_DE_selection_single_{distribution.lower()}_1'
storage_directory = '/home/ahaas/BachelorThesis/hyperparameter'

os.makedirs(storage_directory, exist_ok=True)
db_file_path = os.path.join(storage_directory, f'{study_name}.db')

storage_name = f'sqlite:///{os.path.join(storage_directory, f"{study_name}.db")}'

if not os.path.isfile(db_file_path):
    conn = sqlite3.connect(db_file_path)
    conn.close()

study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)

trials_with_value = [trial for trial in study.trials if trial.value is not None]
try:
    print(f'Trials so far: {len(trials_with_value)}')
    best_params = study.best_params
    print(best_params)
except:
    print('Study not existing')
study.optimize(objective, n_trials=128-len(trials_with_value), show_progress_bar=True, n_jobs=8)
best_params = study.best_params
print(best_params)
best_trial = study.best_trial


