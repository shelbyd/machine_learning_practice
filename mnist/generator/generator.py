import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, ZeroPadding2D, Input, Reshape, UpSampling2D, BatchNormalization, LeakyReLU

from checkpointer import generate_checkpoint_path

noise_input = Input(shape=(128,), name='noise_input')

CHANNELS = 128
INITIAL_WIDTH = 7
INITIAL_HEIGHT = 7

if keras.backend.image_data_format() == 'channels_first':
  reshape = (CHANNELS, INITIAL_HEIGHT, INITIAL_WIDTH)
else:
  reshape = (INITIAL_HEIGHT, INITIAL_WIDTH, CHANNELS)

x = noise_input
x = Dense(128)(x)
x = LeakyReLU()(x)

x = Dense(CHANNELS * INITIAL_HEIGHT * INITIAL_WIDTH)(x)
x = LeakyReLU()(x)
x = Reshape(reshape)(x)

x = UpSampling2D()(x)
x = Conv2D(int(CHANNELS / 2), 5, padding='same')(x)
x = LeakyReLU()(x)

x = UpSampling2D()(x)
x = Conv2D(int(CHANNELS / 4), 5, padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(1, 3, padding='same')(x)
x = Activation('sigmoid')(x)

image = x

generator = Model(noise_input, image, name='generator')

checkpoint_path = generate_checkpoint_path('models/mnist/generator/generator', generator)

try:
  generator.load_weights(checkpoint_path)
except IOError:
  pass
