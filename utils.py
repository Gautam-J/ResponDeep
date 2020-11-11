import os
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
