from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28


def load(f):
    return np.load(f)


# Load the data_npy
x_train = load('data_npy/kmnist-train-imgs/arr_0.npy')
y_train = load('data_npy/kmnist-train-labels/arr_0.npy')
x_test = load('data_npy/kmnist-test-imgs/arr_0.npy')
y_test = load('data_npy/kmnist-test-labels/arr_0.npy')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('{} train samples, {} test samples'.format(len(x_train), len(x_test)))

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.layers.Reshape(input_shape=(28 * 28,), target_shape=(28, 28, 1)),

        keras.layers.Conv2D(kernel_size=3, filters=12, use_bias=False, padding='same'),
        keras.layers.BatchNormalization(center=True, scale=False),
        keras.layers.Activation('relu'),

        keras.layers.Conv2D(kernel_size=6, filters=24, use_bias=False, padding='same', strides=2),
        keras.layers.BatchNormalization(center=True, scale=False),
        keras.layers.Activation('relu'),

        keras.layers.Conv2D(kernel_size=6, filters=32, use_bias=False, padding='same', strides=2),
        keras.layers.BatchNormalization(center=True, scale=False),
        keras.layers.Activation('relu'),

        keras.layers.Flatten(),

        keras.layers.Dense(200, use_bias=False),
        keras.layers.BatchNormalization(center=True, scale=False),
        keras.layers.Activation('relu'),

        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

train_score = model.evaluate(x_train, y_train, verbose=0)
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
