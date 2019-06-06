from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Activation, Dropout
from keras.layers import Dense

from dataProcessor import dp_main
# Aimed to solve galaxy zoo classification problem
def make_network():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (106, 106, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(37))
    model.add(Activation('sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

# the model so far outputs 3D feature maps (height, width, features)
def main():
    # process images
        # make array of image data and array of labelled vectors (one-hot encoding)
        # make sure that each image has the appropriate dimensions
    numPts = 10000 # max of 61578
    images, labels = dp_main(numPts)
    model = make_network()
    model.fit(images, labels, validation_split = 0.2, batch_size = 50, epochs = 20, verbose = 1)
main()
