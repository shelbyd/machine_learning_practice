import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D

import discriminator
from classifier import classifier
import generator

from PIL import Image

(real_images, labels_train), (image_test, labels_test) = mnist.load_data()

def normalize_image(image):
  items, height, width = image.shape
  if keras.backend.image_data_format() == 'channels_first':
    image = image.reshape(items, 1, height, width)
  else:
    image = image.reshape(items, height, width, 1)
  return image.astype('float32') / 255

def denormalize_image(image):
  image *= 255
  image = image.astype(np.uint8)
  if keras.backend.image_data_format() == 'channels_first':
    items, _, height, width = image.shape
  else:
    items, height, width, _ = image.shape
  return image.reshape(items, height, width)

real_images = normalize_image(real_images)
image_test = normalize_image(image_test)

labels_train = keras.utils.to_categorical(labels_train)
labels_test = keras.utils.to_categorical(labels_test)

TRAINING_SIZE = 4096

BATCH_SIZE = 128
EPOCHS = 32

discriminator_on_generator = discriminator.discriminator(generator.image)
classifier_on_generator = classifier(generator.image)
full_generator = Model([generator.digit_input, generator.noise_input],
                       [discriminator_on_generator, classifier_on_generator])
full_generator.compile(
    loss=[
      keras.losses.binary_crossentropy,
      keras.losses.categorical_crossentropy,
    ],
    optimizer='SGD',
)

def random_generator_input(size):
  noise = np.random.uniform(size=(size, generator.noise_input.shape[1]))
  digit = keras.utils.to_categorical(np.random.random_integers(0, 9, (size, 1)))

  return [digit, noise]

def train_discriminator():
  training_size = TRAINING_SIZE / 2
  generated_images = generator.generator.predict(
                         random_generator_input(training_size),
                         batch_size=BATCH_SIZE)

  sampled_images = real_images[np.random.randint(real_images.shape[0], size=training_size)]
  images = np.concatenate((sampled_images, generated_images))
  labels = [1] * training_size + [0] * training_size

  discriminator.trainable = True
  discriminator.discriminator.fit(
      images,
      labels,
      batch_size=BATCH_SIZE,
      callbacks=[keras.callbacks.ModelCheckpoint(discriminator.checkpoint_path)])

def train_generator():
  [digit, noise] = random_generator_input(TRAINING_SIZE)
  discriminator_ones = np.ones((TRAINING_SIZE, 1))

  discriminator.trainable = False
  full_generator.fit([digit, noise],
                     [discriminator_ones, digit],
                     batch_size=BATCH_SIZE,
                     callbacks=[keras.callbacks.ModelCheckpoint(generator.checkpoint_path)])

for epoch in range(EPOCHS):
  train_discriminator()
  train_generator()
