import keras

classifier = keras.models.load_model('models/mnist/cnn.h5')
classifier.name = 'classifier'
classifier.trainable = False
