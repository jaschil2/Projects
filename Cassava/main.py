import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TerminateOnNaN, CSVLogger, ReduceLROnPlateau, EarlyStopping
from models import xception
import numpy as np
import os
from datetime import date

LABEL_FILE = os.path.join(os.getcwd(),'data','train.csv')
IMAGE_DIR = os.path.join(os.getcwd(),'data','train_images')
RUNS_DIR = os.path.join(os.getcwd(),'runs')
RANDOM_SEED = 12345
TARGET_SIZE = (256,256)
NCLASSES = 5
BATCH_SIZE = 32
NEPOCHS = 50

IMAGE_GEN_ARGS = {
			'rotation_range': 60,
			'width_shift_range': 0.3,
			'height_shift_range': 0.3,
			'brightness_range': [0.5,1.5],
			'shear_range': 0.2,
			'zoom_range': 0.2,
			'horizontal_flip': True,
			'rescale': 1/255.,
			'validation_split': 0.2,
			'dtype': np.float32
		}

NGPUS = len(tf.config.list_physical_devices('GPU'))

if NGPUS > 1:
	BATCH_SIZE *= NGPUS
	strategy = tf.distribute.MirroredStrategy()
	with strategy.scope():
		model = xception.build_xception(output_size=NCLASSES,input_size=TARGET_SIZE+(3,))
		optimizer = tf.keras.optimizers.SGD(learning_rate=0.004,momentum=0.9)
		model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
else:
	model = xception.build_xception(output_size=NCLASSES,input_size=TARGET_SIZE+(3,))
	optimizer = tf.keras.optimizers.SGD(learning_rate=0.004,momentum=0.9)
	model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

image_generator = ImageDataGenerator(IMAGE_GEN_ARGS)

train_generator = image_generator.flow_from_directory(
	IMAGE_DIR,
	target_size=TARGET_SIZE,
	batch_size=BATCH_SIZE,
	class_mode='categorical',
	subset='training',
	seed = RANDOM_SEED
	)

validation_generator = image_generator.flow_from_directory(
	IMAGE_DIR,
	target_size=TARGET_SIZE,
	batch_size=BATCH_SIZE,
	class_mode='categorical',
	subset='validation',
	seed = RANDOM_SEED
	)

today = date.today()
date_str = today.strftime("%m-%d-%Y")

run = 0
for f in os.listdir(RUNS_DIR):
	if date_str in f:
		run = max(int(f.split('_')[1]),run)

run_dir = os.path.join(RUNS_DIR,'%s_%d'%(date_str,run+1))
os.mkdir(run_dir)

with open(os.path.join(run_dir,'params.txt'),'w') as f:
	f.write('RANDOM_SEED %s'%RANDOM_SEED)
	f.write('TARGET_SIZE %s'%str(TARGET_SIZE))
	f.write('BATCH_SIZE %s'%BATCH_SIZE)
	f.write('NEPOCHS %s'%NEPOCHS)
	f.write('IMAGE_GEN_ARGS %s'%IMAGE_GEN_ARGS)

csv_logger = CSVLogger(os.path.join(run_dir,'log.csv'))
nan_callback = TerminateOnNaN()

model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs = NEPOCHS,
    callbacks=[nan_callback,csv_logger])