import os
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D

import discriminator
from classifier import classifier
import generator

from PIL import Image

(real_images, real_image_labels), (image_test, labels_test) = mnist.load_data()

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

real_image_labels = keras.utils.to_categorical(real_image_labels)
labels_test = keras.utils.to_categorical(labels_test)

all_noise = np.random.uniform(size=(real_images.shape[0], generator.noise_input.shape[1]))

adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

BATTLES_PER_EPOCH = 8

BATCH_SIZE = 128

def random_generator_input(size):
  digit = keras.utils.to_categorical(np.random.random_integers(9, size=(size, 1)), num_classes=10)
  noise = np.random.uniform(size=(size, generator.noise_input.shape[1]))

  return [digit, noise]



def with_discriminator_batch(f):
  training_size = BATCH_SIZE / 2
  generated_images = generator.generator.predict(
                         random_generator_input(training_size),
                         batch_size=BATCH_SIZE)

  sampled_images = real_images[np.random.randint(real_images.shape[0], size=training_size)]
  images = np.concatenate((sampled_images, generated_images))
  labels = [1] * training_size + [0] * training_size

  return f(*(images, labels))

def train_discriminator():
  return with_discriminator_batch(discriminator.discriminator.train_on_batch)

def test_discriminator():
  return with_discriminator_batch(discriminator.discriminator.test_on_batch)



def with_generator_batch(f):
  [digit, noise] = random_generator_input(BATCH_SIZE)
  discriminator_ones = np.ones((BATCH_SIZE, 1))

  return f(*([digit, noise], [discriminator_ones, digit]))

def train_generator():
  return with_generator_batch(full_generator.train_on_batch)

def test_generator():
  return with_generator_batch(full_generator.test_on_batch)



import math
def do_training_battle():
  discriminator_loss = test_discriminator()
  [_, generator_real_loss, categorical_loss] = test_generator()

  print "discriminator: %f - realness: %f - categorical: %f" % (discriminator_loss, generator_real_loss, categorical_loss)

  if discriminator_loss > generator_real_loss:
    discriminator_loss = train_discriminator()
  else:
    [_, generator_real_loss, categorical_loss] = train_generator()



def generate_and_save_each(directory):
  print "Saving to %s" % directory
  digit = keras.utils.to_categorical([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  noise = np.random.uniform(size=(digit.shape[0], generator.noise_input.shape[1]))

  generated_image = generator.generator.predict([digit, noise])
  denormalized_image = denormalize_image(generated_image)
  target = digit.argmax(0)
  classified = classifier.predict(generated_image).argmax(0)

  try:
    os.makedirs(directory)
  except:
    pass

  for index in range(generated_image.shape[0]):
    image = Image.fromarray(denormalized_image[index])

    image.save("%s/target-%d_classified-%d.png" %
               (directory, target[index], classified[index]))



from datetime import datetime
prefix = datetime.now().isoformat()

generator.generator.trainable = False
discriminator.trainable = True
discriminator.discriminator.compile(loss='binary_crossentropy', optimizer=adam)

generator.generator.trainable = True
discriminator.trainable = False
discriminator_on_generator = discriminator.discriminator(generator.image)
classifier_on_generator = classifier(generator.image)
full_generator = Model([generator.digit_input, generator.noise_input],
                       [discriminator_on_generator, classifier_on_generator])
full_generator.compile(
  loss=[
    keras.losses.binary_crossentropy,
    keras.losses.categorical_crossentropy,
  ],
  optimizer=adam,
)

import sys
for epoch in xrange(sys.maxint):
  generate_and_save_each("/tmp/mnist_images/latest")

  for _ in xrange(BATTLES_PER_EPOCH):
    do_training_battle()

  discriminator.discriminator.save(discriminator.checkpoint_path)
  generator.generator.save(generator.checkpoint_path)
