import os
import tensorflow as tf
import kerastuner as kt
import numpy as np
from model import nnWENO, Const51stOrder, nnWENObeta, nnWENOkt
from tensorflow.keras.callbacks import TerminateOnNaN, CSVLogger, ReduceLROnPlateau, EarlyStopping

RUN_DIR = os.path.join('runs')
XPATH = os.path.join('data','2ndNewAvgs.csv')
YPATH = os.path.join('data','2ndNewFlux.csv')

NEPOCHS = 30
SEED = 123454321

# Loading data

X = np.genfromtxt(XPATH,delimiter=',',autostrip=True,dtype=np.float32).T
Y = np.genfromtxt(YPATH,delimiter=',',autostrip=True,dtype=np.float32)

# Shuffling data in case there are correlations between consecutive rows

np.random.seed(SEED)
idxs = np.arange(X.shape[0])
X = X[idxs,:]
Y = Y[idxs]

# Building callbacks

csv_logger = CSVLogger(os.path.join(RUN_DIR,'log.csv'))
nan_callback = TerminateOnNaN()
reduce_lr = ReduceLROnPlateau(monitor='loss',patience=3,verbose=1)
early_stop = EarlyStopping(monitor='loss',patience=5,min_delta=0.001,verbose=1)

# Running hyperparameter search

tuner = kt.Hyperband(
    nnWENOkt,
    objective='val_loss',
    max_epochs=50,
    seed=SEED,
    hyperband_iterations=2,
    directory='tuning_runs',
    project_name='weno')

tuner.search(X,Y,
	epochs=NEPOCHS,
	validation_split=0.1,
	callbacks=[nan_callback,csv_logger,reduce_lr,early_stop])
