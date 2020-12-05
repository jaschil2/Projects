import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Input, add, Lambda, SeparableConv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras import datasets

class SeparableBlock:
	def __init__(self,channels,activate_first=True):
		self.channels=channels
		self.activate_first=activate_first

	def __call__(self,x):
		if self.activate_first:
			z = relu(x)
		else:
			z = x
		if isinstance(self.channels,list) or isinstance(self.channels,tuple):
			channels_1, channels_2 = self.channels
		else:
			channels_1, channels_2 = self.channels, self.channels

		z = SeparableConv2D(
			filters=channels_1,
			kernel_size=3,
			strides=1,
			use_bias=False,
			padding='same')(z)
		z = BatchNormalization()(z)
		z = relu(z)
		z = SeparableConv2D(
			filters=channels_2,
			kernel_size=3,
			strides=1,
			use_bias=False,
			padding='same')(z)
		z = MaxPool2D(
			pool_size=(3,3),
			strides=2,
			padding='same')(z)

		r = Conv2D(
			filters=channels_2,
			kernel_size=1,
			strides=2,
			use_bias=False,
			padding='same')(x)
		r = BatchNormalization()(r)

		return add([z,r])


def build_xception(output_size,input_size=(299,299,3)):
	# Entry flow
	input_layer = Input(shape=input_size)
	x = Conv2D(
		filters=32,
		kernel_size=3,
		strides=2,
		use_bias=False)(input_layer)
	x = BatchNormalization()(x)
	x = relu(x)
	x = Conv2D(
		filters=64,
		kernel_size=3,
		strides=1,
		use_bias=False)(x)
	x = BatchNormalization()(x)
	x = relu(x)
	x = SeparableBlock(channels=128,activate_first=False)(x)
	x = SeparableBlock(channels=256)(x)
	r = SeparableBlock(728)(x)

	# Middle flow
	x = relu(x)
	x = SeparableConv2D(
		filters=728,
		kernel_size=3,
		strides=1,
		use_bias=False,
		padding='same')(r)
	x + BatchNormalization()(x)
	x = relu(x)
	x = SeparableConv2D(
		filters=728,
		kernel_size=3,
		strides=1,
		use_bias=False,
		padding='same')(x)
	x = BatchNormalization()(x)
	x = relu(x)
	x = SeparableConv2D(
		filters=728,
		kernel_size=3,
		strides=1,
		use_bias=False,
		padding='same')(x)
	x = BatchNormalization()(x)
	x = add([x,r])

	# Exit flow
	x = SeparableBlock(channels=(728,1024))(x)
	x = SeparableConv2D(
		filters=1536,
		kernel_size=3,
		strides=1,
		use_bias=False,
		padding='same')(x)
	x = BatchNormalization()(x)
	x = relu(x)
	x = SeparableConv2D(
		filters=2048,
		kernel_size=3,
		strides=1,
		use_bias=False,
		padding='same')(x)
	x = BatchNormalization()(x)
	x = relu(x)
	x = GlobalAveragePooling2D()(x)
	x = Dense(256,activation='sigmoid')(x)
	x = Dense(256,activation='sigmoid')(x)
	output_layer = Dense(output_size,activation='sigmoid')(x)

	model = Model(input_layer,output_layer)

	return model


