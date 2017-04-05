import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, Input, Flatten

from checkpointer import generate_checkpoint_path

if keras.backend.image_data_format() == 'channels_first':
  input_shape = (1, 28, 28)
else:
  input_shape = (28, 28, 1)

image_input = Input(shape=input_shape)
x = image_input

x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)

x = Conv2D(64, (2, 2), strides=(2, 2), padding='same')(x)
x = Activation('relu')(x)

x = Conv2D(64, (2, 2), strides=(2, 2), padding='same')(x)
x = Activation('relu')(x)

x = Dropout(0.2)(x)
x = Flatten()(x)

x = Dense(256)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(1)(x)
x = Activation('sigmoid', name='discriminator_is_real')(x)

is_real = x

discriminator = Model(image_input, is_real, name='discriminator')

checkpoint_path = generate_checkpoint_path('models/mnist/generator/discriminator', discriminator)

try:
  discriminator.load_weights(checkpoint_path)
except IOError:
  pass

_optimizer = keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
discriminator.compile(loss='binary_crossentropy', optimizer=_optimizer)
