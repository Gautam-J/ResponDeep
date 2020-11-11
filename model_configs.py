from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (EarlyStopping,
                                        ModelCheckpoint,
                                        ReduceLROnPlateau)


def buildModel(input_shape, n_classes):
    baseModel = MobileNetV2(input_shape=input_shape,
                            weights='imagenet',
                            include_top=False,
                            pooling='avg')

    baseModel.trainable = False

    inputs = Input(input_shape)
    x = baseModel(inputs)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
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
