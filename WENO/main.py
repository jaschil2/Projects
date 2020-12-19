import os
import tensorflow as tf
import numpy as np
from model import nnWENO, Const51stOrder, nnWENObeta
from tensorflow.keras.callbacks import TerminateOnNaN, CSVLogger, ReduceLROnPlateau, EarlyStopping

# TODO: add argparser to get rid of some of this stuff

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

# Selecting model

# nn_model = nnWENO(hidden_channels=[64,64,64,64])
nn_model = Const51stOrder(0.1)
# nn_model = nnWENObeta(hidden_channels=[64,64,64,64])

nn_model.summary() # print architecture

# Compile model

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
nn_model.compile(optimizer=optimizer,
	loss='mse',
	metrics=[tf.keras.metrics.MeanAbsoluteError()])

# Building callbacks

csv_logger = CSVLogger(os.path.join(RUN_DIR,'log.csv'))
nan_callback = TerminateOnNaN()
reduce_lr = ReduceLROnPlateau(monitor='loss',patience=3,verbose=1)
early_stop = EarlyStopping(monitor='loss',patience=5,min_delta=0.005,verbose=1)

# Fit model

nn_model.fit(X,Y,
	epochs=NEPOCHS,
	validation_split=0.1,
	callbacks=[nan_callback,csv_logger,reduce_lr,early_stop])
