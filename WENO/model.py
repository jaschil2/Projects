import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

def nnWENOkt(hp):
	# keras-tuner compatible model, TODO: build method to instantiate model from optimal params
	inp = Input(shape=(5,))

	w = inp

	# Scale values to interval [0,1]
	def scale_input(x):
		mx = tf.reduce_max(x,axis=1,keepdims=True)
		mn = tf.reduce_min(x,axis=1,keepdims=True)
		L = mx-mn
		return tf.math.divide_no_nan(x-mn, L)

	w = Lambda(scale_input)(w)

	reg = regularizers.l1(hp.Float(name='l1_penalty',min_value=0,max_value=3,step=0.05)) # Using L1-norm regularization

	# Build main hidden layers
	for layer in range(hp.Int(name='nlayers',min_value=2,max_value=8)):
		w = Dense(hp.Int(name='layer_'+str(layer),min_value=32,max_value=128,step=32),
			use_bias=False,
			kernel_regularizer=reg,
			activation='relu')(w)
		w = BatchNormalization()(w)
		w = Dropout(0.5)(w)

	# Reduce dimensions to 5
	w = Dense(5,
		use_bias=False,
		activation='relu')(w)
	
	# Modify WENO weights by adding NN perturbation and renormalizing
	def modify_weno(x):
		weno = tf.expand_dims(tf.constant([1/30, -13/60, 47/60, 9/20, -1/20]),axis=0)
		weno = x + weno
		tot = tf.reduce_sum(weno,axis=1,keepdims=True)
		return x / tot
	w = Lambda(modify_weno)(w)

	# Apply weights and sum
	out = Dot(axes = 1)([inp,w])

	model = Model(inp,out)

	optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.Float(name='learning_rate',min_value=1e-4,max_value=1e-2,sampling='log'))
	model.compile(optimizer=optimizer,
		loss='mse',
		metrics=[tf.keras.metrics.MeanAbsoluteError()])

	return model

def nnWENO(input_shape=(5,),hidden_channels=[64,64,64,64],batch_normalize=True,dropout=True):

	inp = Input(shape=input_shape)

	w = inp
	def scale_input(x):
		mx = tf.reduce_max(x,axis=1,keepdims=True)
		mn = tf.reduce_min(x,axis=1,keepdims=True)
		L = mx-mn
		return tf.math.divide_no_nan(x-mn, L)

	w = Lambda(scale_input)(w)

	for nchan in hidden_channels:
		w = Dense(nchan,activation='relu')(w)
		if batch_normalize:
			w = BatchNormalization()(w)
		if dropout:
			w = Dropout(0.5)(w)

	w = Dense(5,activation='relu')(w)
	
	def modify_weno(x):
		# shape = tf.shape(x)
		# weno = tf.broadcast_to(tf.constant([1/30, -13/60, 47/60, 9/20, -1/20]),shape)
		weno = tf.expand_dims(tf.constant([1/30, -13/60, 47/60, 9/20, -1/20]),axis=0)
		weno = x + weno
		tot = tf.reduce_sum(weno,axis=1,keepdims=True)
		return x / tot
	w = Lambda(modify_weno)(w)

	out = Dot(axes = 1)([inp,w])

	model = Model(inp,out)

	return model

def nnWENObeta(input_shape=(5,),hidden_channels=[64,64,64,64],batch_normalize=True,dropout=True):

	inp = Input(shape=input_shape)

	w = inp
	def scale_input(x):
		mx = tf.reduce_max(x,axis=1,keepdims=True)
		mn = tf.reduce_min(x,axis=1,keepdims=True)
		L = mx-mn
		return tf.math.divide_no_nan(x-mn, L)

	w = Lambda(scale_input)(w)

	for nchan in hidden_channels:
		w = Dense(nchan,activation='relu')(w)
		if batch_normalize:
			w = BatchNormalization()(w)
		if dropout:
			w = Dropout(0.5)(w)

	w = Dense(3,activation='relu')(w)

	def normalize(x):
		tot = tf.reduce_sum(x,axis=1,keepdims=True)
		return x / tot
	
	def calc_beta(x):
		eps = tf.constant(1e-6)
		b1 = (4*x[:,0]**2 - 19*x[:,0]*x[:,1] + 25*x[:,1]**2 + 11*x[:,0]*x[:,2] - 31*x[:,1]*x[:,2] + 10*x[:,2]**2)/3
		b1 = tf.expand_dims(b1,-1)
		b2 = (4*x[:,1]**2 - 13*x[:,1]*x[:,2] + 13*x[:,2]**2 + 5*x[:,1]*x[:,3] - 13*x[:,2]*x[:,3] + 4*x[:,3]**2)/3
		b2 = tf.expand_dims(b2,-1)
		b3 = (10*x[:,2]**2 - 31*x[:,2]*x[:,3] + 25*x[:,3]**2 + 11*x[:,2]*x[:,4] - 19*x[:,3]*x[:,4] + 4*x[:,4]**2)/3
		b3 = tf.expand_dims(b3,-1)

		beta = Concatenate()([b1,b2,b3])

		weights = tf.constant([1/16,5/8,5/16])
		weights = tf.expand_dims(weights,axis=0)

		weights = weights / (beta + eps)**2
		
		return weights
		
	weights = Lambda(calc_beta)(inp)
	weights = add([weights,w])

	weights = Lambda(normalize)(weights)

	def calc_output(x):
		u1 = tf.expand_dims((3/8)*x[:,0] - (5/2)*x[:,1] + (15/8)*x[:,2],-1)
		u2 = tf.expand_dims((-1/8)*x[:,1] + (3/4)*x[:,2] + (3/8)*x[:,3],-1)
		u3 = tf.expand_dims((3/8)*x[:,2] + (3/4)*x[:,3] - (1/8)*x[:,4],-1)
		return Concatenate()([u1,u2,u3])

	out = Lambda(calc_output)(inp)
	out = Dot(axes = 1)([out,weights])

	model = Model(inp,out)

	return model

def Const51stOrder(regC):
    pntsuse = 5

    H51 = np.array([[0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0]])
    H51c = np.array([1/30, -13/60, 47/60, 9/20, -1/20])
        
    wub1 = np.array([[-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5]])
    wub1c = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
    
    
    # Make weights for the projection
    u_05 = Input(shape = (5, ))#merge all the average inputs as u_(-2),u_(-1),u_0,u_1,u_2,u_3
        
    Cs = Dense(5,trainable=False,weights=[H51,H51c])(u_05)#Final WENO5 coefficients
    reggersA = 0.001
    reggersb = 0.001
    
    x1 = Dense(5,activation='relu')(u_05)
    x2 = Dense(5,activation='relu')(x1)
    x3 = Dense(5,activation='relu')(x2)

    #TODO: Pass arguments to this function that define the regularization and neural network nodes/layers and l1/l2 optimization
    #dc = Dense(5,activity_regularizer=regularizers.l2(regC))(x9)#end the DNN, the 5 differences are the outputs
    dc = Dense(5,trainable=False,activity_regularizer=regularizers.l2(regC))(x3)#end the DNN, the 5 differences are the outputs
    c_tilde = Subtract()([Cs,dc])#use the differences to modify the coefficients
    
    dc2 = Dense(pntsuse,trainable=False,weights=[wub1,wub1c])(c_tilde)
    #compute how each coefficient must be changed for consistency
    
    c_all = Add()([c_tilde,dc2])
    
    p2 = dot([u_05,c_all], axes = 1, normalize = False)#compute flux from all 5 coefficients
    #p2 = dot([u_05,Cs], axes = 1, normalize = False)#compute flux from all 5 coefficients
    
    model = Model(inputs=u_05, outputs=[p2])
    return model
