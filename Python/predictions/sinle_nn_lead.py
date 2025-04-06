import pandas as pd
import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from datetime import datetime, timedelta
import tensorflow.compat.v2.keras as keras
import sys
import os
from multiprocessing import Pool
import json
import copy

paramcount = {'Normal': 2,
              'JSU': 4,
              'Point': None}
distribution = 'JSU'
trial = 1

filepath = f'/home/ahaas/BachelorThesis/distparams_singleNN1_{distribution.lower()}_{trial}'
if not os.path.exists(filepath):
    os.mkdir(filepath)

#load hyperparameter
try:
    with open(f'../params_trial_single_{trial}.json', 'r') as j:
        params = json.load(j)
except:
    with open(f'/home/ahaas/BachelorThesis/params_trial_single_{trial}.json', 'r') as j:
        params = json.load(j)

#load data
try:
    data = pd.read_csv('../../Datasets/DE.csv', index_col=0)
except:
    data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)

data.index = pd.to_datetime(data.index)
#train models
def runoneday(inp):
    params, dayno = inp
    fc_period = int(24)
    df = data.iloc[dayno*24:dayno*24+1456*24+fc_period]

    if os.path.exists(os.path.join(filepath, datetime.strftime(df.index[-24], '%Y-%m-%d'))):
        return
    # prepare the input/output dataframes
    Y = df.iloc[:, 0].to_numpy()
    Y = Y[7*24:(1456*24)] # skip first 7 days

    X = np.zeros(((1456*24) + fc_period, 14))
    for d in range(7*24, (1456*24) + fc_period):
        X[d, 0] = df.iloc[(d-1*24), 0] # D-1 price
        X[d, 1] = df.iloc[(d-2*24), 0] # D-2 price
        X[d, 2] = df.iloc[(d-3*24), 0] # D-3 price
        X[d, 3] = df.iloc[(d-7*24), 0] # D-7 price
        X[d, 4] = df.iloc[d, 1] # D load forecast
        X[d, 5] = df.iloc[(d-1*24), 1] # D-1 load forecast
        X[d, 6] = df.iloc[(d-7*24), 1] # D-7 load forecast
        X[d, 7] = df.iloc[d, 2] # D RES sum forecast
        X[d, 8] = df.iloc[(d-1*24), 2] # D-1 RES sum forecast
        X[d, 9] = df.iloc[(d-2*24), 3] # D-2 EUA
        X[d, 10] = df.iloc[(d-2*24), 4] # D-2 API2_Coal
        X[d, 11] = df.iloc[(d-2*24), 5] # D-2 TTF_Gas
        X[d, 12] = df.iloc[(d-2*24), 6] # D-2 Brent oil
        X[d, 13] = df.index[d].weekday()

    # '''
    # input feature selection
    colmask = [False] * 14
    if params['price_D-1']:
        colmask[0] = True
    if params['price_D-2']:
        colmask[1] = True
    if params['price_D-3']:
        colmask[2] = True
    if params['price_D-7']:
        colmask[3] = True
    if params['load_D']:
        colmask[4] = True
    if params['load_D-1']:
        colmask[5] = True
    if params['load_D-7']:
        colmask[6] = True
    if params['RES_D']:
        colmask[7] = True
    if params['RES_D-1']:
        colmask[8] = True
    if params['EUA']:
        colmask[9] = True
    if params['Coal']:
        colmask[10] = True
    if params['Gas']:
        colmask[11] = True
    if params['Oil']:
        colmask[12] = True
    if params['Dummy']:
        colmask[13] = True

    X = X[:, colmask]
    Xf = X[-fc_period:, :]
    X = X[(7*24):-fc_period, :]

    fcs = {k: [] for k in ['loc', 'scale', 'tailweight', 'skewness']}
    for tm in range(24):
        X_tm = X[tm::24]
        Y_tm = Y[tm::24]
        Xf_tm = Xf[tm::24]


        inputs = keras.Input(X_tm.shape[1])
        # batch normalization
        norm = keras.layers.BatchNormalization()(inputs)
        last_layer = norm

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
                          else params['h1_activation_rate_l1'])
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

        # now define parameter layers with their regularization
        param_layers = []
        param_names = ["loc", "scale", "tailweight", "skewness"]
        for p in range(paramcount[distribution]):
            regularize_param_kernel = params['regularize_'+param_names[p]]
            param_kernel_rate = (0.0 if not regularize_param_kernel
                                 else params[str(param_names[p])+'_rate_l1'])
            param_layers.append(keras.layers.Dense(
                1, activation='linear', # kernel_initializer='ones',
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
                        tailweight=1 + 3 * tf.math.softplus(t[..., 2]),
                        skewness=t[..., 3]))(linear)

        else:
            raise ValueError(f'Incorrect distribution {distribution}')
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']),
                      loss=lambda y, rv_y: -rv_y.log_prob(y),
                      metrics='mae')

        callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
        perm = np.random.permutation(np.arange(X_tm.shape[0]))
        VAL_DATA = .2
        trainsubset = perm[:int((1 - VAL_DATA) * len(perm))]
        valsubset = perm[int((1 - VAL_DATA) * len(perm)):]
        model.fit(X_tm[trainsubset], Y_tm[trainsubset], epochs=1500, validation_data=(X_tm[valsubset], Y_tm[valsubset]),
                  callbacks=callbacks, batch_size=32, verbose=False)

        dist = model(Xf_tm)
        if distribution == 'Normal':
            getters = {'loc': dist.loc, 'scale': dist.scale}
        elif distribution in {'JSU', 'SinhArcsinh', 'NormalInverseGaussian'}:
            getters = {'loc': dist.loc, 'scale': dist.scale,
                       'tailweight': dist.tailweight, 'skewness': dist.skewness}
        for k, v in getters.items():
            fcs[k].append(v.numpy().tolist()[0])
    print(fcs)
    with open(os.path.join(filepath, datetime.strftime(df.index[-24], '%Y-%m-%d')), 'w') as j:
        json.dump(fcs, j)
    print(datetime.strftime(df.index[-24], '%Y-%m-%d'))
    print(datetime.now().strftime('%H:%M:%S'))



#realigning forecasts, for daily predictions
inputlist = [(params, day) for day in range(182, 736)]

#for e in inputlist:
#     _ = runoneday(e)
#print(os.cpu_count())

if __name__ == '__main__':
    with Pool(16) as p:
        _ = p.map(runoneday, inputlist)