import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers, utils
import time

import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

EPOCHS = 5
BATCH_SIZE=128
NUM_CLASSES=10
INPUT_SHAPE = (28,28,1)

def build_model(input_shape, classes):
    model = models.Sequential()
    model.add(layers.Convolution2D(20, (5,5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Convolution2D(50, (5,5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))

    return model


def main():
    if tf.test.gpu_device_name():
        print("GPU device: {}".format(tf.test.gpu_device_name()))
    else:
        print("No GPU found, exiting...")
        return

    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)

    if gpus:
        print("GPU found")
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        
        # disable GPU
        # tf.config.experimental.set_visible_devices([], 'GPU')
    else:
        print("No GPU found, exiting...")
        return

    model = build_model(INPUT_SHAPE, NUM_CLASSES)
    optimizer = optimizers.Adam()

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape((60000,28,28,1))
    x_test = x_test.reshape((10000,28,28,1))

    x_train, x_test = x_train/255.0, x_test/255.0

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = utils.to_categorical(y_train, NUM_CLASSES)
    y_test = utils.to_categorical(y_test, NUM_CLASSES)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    t1 = time.time()
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.8)
    t2 = time.time()
    print("training time = {}".format(t2-t1))

    score=model.evaluate(x_test, y_test)
    print("\nTest score:", score[0])
    print("\nTest accurace", score[1])

if __name__ == '__main__':
    main()


