import os
import cv2
import time


def createBaseDir():
    baseDir = f'models/model_{int(time.time())}'

    if not os.path.exists(baseDir):
        os.makedirs(baseDir)

    return baseDir


def getNumberOfPNGInDirectory(directory):
    return len([i for i in os.listdir(directory) if i.endswith('.png')])


def getNumberOfClasses(path_to_train_dir):
    return len(list(os.listdir(path_to_train_dir)))


def getPathToLatestModel():
    if not os.path.exists('models'):
        print('[ERROR] Models directory does not exists')
        return None

    latestModelDirectory = os.path.join('models', os.listdir('models')[-1])

    for file in os.listdir(latestModelDirectory):
        if file.endswith('.h5'):
            return os.path.join(latestModelDirectory, file)


def getLabelIndices():
    labelIndices = dict()

    with open('labels.txt', 'r') as f:
        lines = f.readlines()

    for line in lines:
        index, label = line.split()
        labelIndices[int(index)] = label

    return labelIndices


def loadComicImage(path_to_image, size=(400, 400)):
    image = cv2.imread(path_to_image, -1)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    if len(image.shape) > 2 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image
