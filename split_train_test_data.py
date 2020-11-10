import os
import argparse
from math import floor
from utils import getNumberOfPNGInDirectory


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=float, default=0.2,
                        help='Percentage of data to use for testing. Defaults to 0.2')

    args = parser.parse_args()

    return args


args = getArgs()
testSize = args.size

for labelName in os.listdir('data'):
    print(f'[INFO] Parsing label {labelName}')
    trainCount = 0
    testCount = 0

    labelDirectory = os.path.join('data', labelName)
    trainDirectory = os.path.join(labelDirectory, 'train')
    testDirectory = os.path.join(labelDirectory, 'test')

    if not os.path.exists(trainDirectory):
        os.makedirs(trainDirectory)

    if not os.path.exists(testDirectory):
        os.makedirs(testDirectory)

    totalNumberOfFiles = getNumberOfPNGInDirectory(labelDirectory)
    testFileMax = floor(testSize * totalNumberOfFiles)

    for fileName in os.listdir(labelDirectory):
        if not fileName.endswith('.png'):
            continue

        originalPathToFile = os.path.join(labelDirectory, fileName)

        if int(fileName.split('.')[0]) < testFileMax:
            newPathToFile = os.path.join(testDirectory, fileName)
            testCount += 1
        else:
            newPathToFile = os.path.join(trainDirectory, fileName)
            trainCount += 1

        os.rename(originalPathToFile, newPathToFile)

    print(f"[DEBUG] {trainCount=} {testCount=}\n")
