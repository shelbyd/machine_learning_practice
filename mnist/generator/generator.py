import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, ZeroPadding2D, Input, Reshape, UpSampling2D

from checkpointer import generate_checkpoint_path

digit_input = Input(shape=(10,), name='digit_input')

noise_input = Input(shape=(100,), name='noise_input')

CHANNELS = 64
INITIAL_WIDTH = 7
INITIAL_HEIGHT = 7

if keras.backend.image_data_format() == 'channels_first':
  reshape = (CHANNELS, INITIAL_HEIGHT, INITIAL_WIDTH)
else:
  reshape = (INITIAL_HEIGHT, INITIAL_WIDTH, CHANNELS)

digit_x = Dense(256)(digit_input)
digit_x = Activation('relu')(digit_x)
digit_x = Dropout(0.5)(digit_x)

x = keras.layers.concatenate([digit_x, noise_input])
x = Dense(256)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(CHANNELS * INITIAL_HEIGHT * INITIAL_WIDTH)(x)
x = Activation('relu')(x)
x = Reshape(reshape)(x)
x = UpSampling2D()(x)
x = Conv2D(CHANNELS, 3, padding='same')(x)
x = Activation('relu')(x)
x = UpSampling2D()(x)
x = Conv2D(1, 2, padding='same')(x)
x = Activation('relu')(x)

image = x

generator = Model([digit_input, noise_input], image, name='generator')

checkpoint_path = generate_checkpoint_path('models/mnist/generator/generator', generator)

try:
  generator.load_weights(checkpoint_path)
except IOError:
  pass
