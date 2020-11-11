from tensorflow.keras.preprocessing.image import ImageDataGenerator


def getTrainDatagen():
    return ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1 / 255.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2)


def getTestDatagen():
    return ImageDataGenerator(rescale=1 / 255.)


def getTrainGenerator(directory, target_size, batch_size):
    global trainDatagen
    trainDatagen = getTrainDatagen()

    print('[INFO] Train Generator')
    trainGenerator = trainDatagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset='training')

    return trainGenerator


def getValGenerator(directory, target_size, batch_size):

    print('[INFO] Validation Generator')
    valGenerator = trainDatagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset='validation')

    return valGenerator


def getTestGenerator(directory, target_size, batch_size):
    testDatagen = getTestDatagen()

    print('[INFO] Test Generator')
    testGenerator = testDatagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size)

    return testGenerator
