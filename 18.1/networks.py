from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

# All models as described in http://cs231n.stanford.edu/reports/2016/pdfs/300_Report.pdf.


def _simple_model(p, kernel_size, dense_size=128):
    # Model Preparation
    model = Sequential()

    # First layer of convolutions
    model.add(BatchNormalization(input_shape=(33, 33, 1)))
    model.add(Conv2D(filters=32, kernel_size=kernel_size, activation='relu',
                     padding='same', kernel_initializer='he_uniform'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Second layer of convolutions
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same', kernel_initializer='he_uniform'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(p))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='he_uniform'))
    opt = Adam(lr=0.00025)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _lanet(p, kernel_size, dense_size=128):
    # Model Preparation
    model = Sequential()

    model.add(BatchNormalization(input_shape=(33, 33, 1)))

    # First layer of convolutions
    model.add(Conv2D(filters=32, kernel_size=kernel_size, activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Second layer of convolutions
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Third layer of convolutions
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Fourth layer of convolutions
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(dense_size * 4, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(p))
    model.add(Dense(dense_size * 4, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(p))
    model.add(Dense(dense_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(p))
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=0.00005)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _lanet2(p, kernel_size, dense_size=128):
    # Model Preparation
    model = Sequential()

    model.add(BatchNormalization(input_shape=(33, 33, 1)))

    # First layer of convolutions
    model.add(Conv2D(filters=32, kernel_size=kernel_size, activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Second layer of convolutions
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Third layer of convolutions
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Fourth layer of convolutions
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Fifth layer of convolutions
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(dense_size * 4, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(p))
    model.add(Dense(dense_size * 4, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(p))
    model.add(Dense(dense_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(p))
    model.add(Dense(1, activation='sigmoid'))


    # opt = Adam(lr=0.00005)
    # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _lanet3(p, kernel_size, dense_size=128):
    # Model Preparation
    model = Sequential()

    model.add(BatchNormalization(input_shape=(33, 33, 1)))

    # First layer of convolutions
    model.add(Conv2D(filters=64, kernel_size=kernel_size, activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Second layer of convolutions
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Third layer of convolutions
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Fourth layer of convolutions
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(p))
    model.add(Dense(dense_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(p))
    model.add(Dense(dense_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(p))
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=0.00005)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Returns either Simple Model, or one of 3 LaNETs.
# Case invariant.
def get_model(name, p, kernel_size, dense_size=128):
    assert type(name) is str
    name = str.lower(name)
    if name == 'simple model' or \
            name == 'sm' or name == 'simple':
        return _simple_model(p, kernel_size, dense_size)
    elif name == 'lanet':
        return _lanet(p, kernel_size, dense_size)
    elif name == 'lanet2':
        return _lanet2(p, kernel_size, dense_size)
    elif name == 'lanet3':
        return _lanet3(p, kernel_size, dense_size)

    raise ValueError()
