import os
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from keras import backend as K

import discriminator
from classifier import classifier
import generator

from PIL import Image

(real_images, _), _ = mnist.load_data()

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

BATTLES_PER_EPOCH = 8

BATCH_SIZE = 128

def random_generator_input(size):
  return np.random.normal(size=(size, generator.noise_input.shape[1]))


def with_discriminator_batch(f):
  training_size = BATCH_SIZE / 2
  generated_images = generator.generator.predict(
                         random_generator_input(training_size),
                         batch_size=BATCH_SIZE)

  sampled_images = real_images[np.random.randint(real_images.shape[0], size=training_size)]
  images = np.concatenate((sampled_images, generated_images))
  labels = [1] * sampled_images.shape[0] + [0] * generated_images.shape[0]

  return f(*(images, labels))

def train_discriminator():
  return with_discriminator_batch(discriminator.discriminator.train_on_batch)

def test_discriminator():
  return with_discriminator_batch(discriminator.discriminator.test_on_batch)



def with_generator_batch(f):
  noise = random_generator_input(BATCH_SIZE)
  discriminator_ones = np.ones((BATCH_SIZE, 1))

  return f(*(noise, discriminator_ones))

def train_generator():
  return with_generator_batch(full_generator.train_on_batch)

def test_generator():
  return with_generator_batch(full_generator.test_on_batch)



def symmetric_training():
  discriminator_loss = train_discriminator()
  generator_real_loss = train_generator()

  print "discriminator: %f - realness: %f" % (discriminator_loss, generator_real_loss)

def one_sided_training():
  discriminator_loss = test_discriminator()
  generator_real_loss = test_generator()

  if discriminator_loss > generator_real_loss:
    discriminator_loss = train_discriminator()
    print "discriminator: %f - realness: %f - training: discriminator" % (discriminator_loss, generator_real_loss)
  else:
    generator_real_loss = train_generator()
    print "discriminator: %f - realness: %f - training: generator" % (discriminator_loss, generator_real_loss)



def generate_and_save_each(directory):
  print "Saving to %s" % directory
  noise = random_generator_input(10)

  generated_image = generator.generator.predict(noise)
  denormalized_image = denormalize_image(generated_image)
  classified = classifier.predict(generated_image).argmax(0)

  try:
    os.makedirs(directory)
  except:
    pass

  for index in range(generated_image.shape[0]):
    image = Image.fromarray(denormalized_image[index])

    image.save("%s/%d_classified-%d.png" %
               (directory, index, classified[index]))



generator.generator.trainable = False
discriminator.discriminator.trainable = True
discriminator.discriminator.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Nadam(lr=0.0002))

generator.generator.trainable = True
discriminator.discriminator.trainable = False
discriminator_on_generator = discriminator.discriminator(generator.image)
full_generator = Model(generator.noise_input,
                       discriminator_on_generator)
full_generator.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Nadam(lr=0.0002))

discriminator.discriminator.summary()
generator.generator.summary()

import sys
while True:
  generate_and_save_each("/tmp/mnist_images/latest")

  for _ in xrange(BATTLES_PER_EPOCH):
    one_sided_training()

  discriminator.discriminator.save(discriminator.checkpoint_path)
  generator.generator.save(generator.checkpoint_path)
