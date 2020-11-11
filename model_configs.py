from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, ReLU, GlobalMaxPool2D
from tensorflow.keras.callbacks import (EarlyStopping,
                                        ModelCheckpoint,
                                        ReduceLROnPlateau)


def buildModel(input_shape, n_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), strides=1, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Conv2D(128, (3, 3), strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Dropout(0.2))

    model.add(GlobalMaxPool2D())

    model.add(Dense(128))
    model.add(ReLU())

    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))

    model.summary()

    return model


def getModelCallbacks(directory):
    filePath = '{epoch:02d}_{val_loss:.4f}.hdf5'

    modelCheckpoint = ModelCheckpoint(f'{directory}/{filePath}',
                                      monitor='val_loss',
                                      save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5,
                                  restore_best_weights=True)

    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                 patience=2, min_lr=1e-6)

    return [modelCheckpoint,
            reduceLR,
            earlyStopping]
