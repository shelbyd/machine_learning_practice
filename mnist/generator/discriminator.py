import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, Input, Flatten, MaxPooling2D, LeakyReLU

from checkpointer import generate_checkpoint_path

if keras.backend.image_data_format() == 'channels_first':
  input_shape = (1, 28, 28)
else:
  input_shape = (28, 28, 1)

image_input = Input(shape=input_shape)
x = image_input

x = Conv2D(64, 5, padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(64, 5, strides=(2, 2), padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(32, 5, strides=(2, 2), padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(16, 5, strides=(2, 2), padding='same')(x)
x = LeakyReLU()(x)

x = Flatten()(x)
x = Dense(128)(x)
x = Dropout(0.5)(x)
x = LeakyReLU()(x)

x = Dense(1)(x)
x = Activation('sigmoid', name='discriminator_is_real')(x)

is_real = x

discriminator = Model(image_input, is_real, name='discriminator')

checkpoint_path = generate_checkpoint_path('models/mnist/generator/discriminator', discriminator)

try:
  discriminator.load_weights(checkpoint_path)
except IOError:
  pass
