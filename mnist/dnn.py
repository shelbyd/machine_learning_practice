import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop

number_of_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def normalize_image(image):
    items, height, width = image.shape
    return image.reshape(items, width * height).astype('float32') / 255

x_train = normalize_image(x_train)
x_test = normalize_image(x_test)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'])

history = model.fit(x_train, y_train,
        batch_size=128,
        epochs=20,
        verbose=1,
        validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
