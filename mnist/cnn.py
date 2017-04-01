import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D

try:
  classifier = keras.models.load_model('models/mnist/cnn.h5')
except:
  classifier = Sequential()

  classifier.add(ZeroPadding2D(input_shape=input_shape))
  classifier.add(Conv2D(32, (3, 3)))
  classifier.add(Activation('relu'))

  classifier.add(ZeroPadding2D())
  classifier.add(Conv2D(64, (3, 3)))
  classifier.add(Activation('relu'))

  classifier.add(Conv2D(64, (2, 2), strides=(2, 2)))
  classifier.add(Activation('relu'))

  classifier.add(Dropout(0.2))
  classifier.add(Flatten())

  classifier.add(Dense(128))
  classifier.add(Activation('relu'))
  classifier.add(Dropout(0.5))

  classifier.add(Dense(y_train.shape[1]))
  classifier.add(Activation('softmax'))

if __name__ == "__main__":
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  def normalize_image(image):
    items, height, width = image.shape
    if keras.backend.image_data_format() == 'channels_first':
      image = image.reshape(items, 1, height, width)
    else:
      image = image.reshape(items, height, width, 1)
    return image.astype('float32') / 255

  x_train = normalize_image(x_train)
  x_test = normalize_image(x_test)

  y_train = keras.utils.to_categorical(y_train)
  y_test = keras.utils.to_categorical(y_test)

  if keras.backend.image_data_format() == 'channels_first':
    input_shape = (1, x_train.shape[1], x_train.shape[2])
  else:
    input_shape = (x_train.shape[1], x_train.shape[2], 1)


  classifier.summary()

  classifier.compile(loss='categorical_crossentropy',
          optimizer=keras.optimizers.Adadelta(),
          metrics=['accuracy'])

  history = classifier.fit(x_train, y_train,
          batch_size=128,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[keras.callbacks.ModelCheckpoint('models/mnist/cnn.h5', verbose=1)])

  score = classifier.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
