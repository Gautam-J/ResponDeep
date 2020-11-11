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

trainDirectory = os.path.join('data', 'train')
validationDirectory = os.path.join('data', 'validation')

if not os.path.exists(trainDirectory):
    os.makedirs(trainDirectory)

if not os.path.exists(validationDirectory):
    os.makedirs(validationDirectory)

for labelName in os.listdir('data'):
    if labelName in ['train', 'validation']:
        continue

    print(f'[INFO] Parsing label {labelName}')
    trainCount = 0
    testCount = 0

    labelDirectory = os.path.join('data', labelName)
    totalNumberOfFiles = getNumberOfPNGInDirectory(labelDirectory)
    testFileMax = floor(testSize * totalNumberOfFiles)

    labelTrainDirectory = os.path.join(trainDirectory, labelName)
    labelValidationDirectory = os.path.join(validationDirectory, labelName)

    if not os.path.exists(labelTrainDirectory):
        os.makedirs(labelTrainDirectory)

    if not os.path.exists(labelValidationDirectory):
        os.makedirs(labelValidationDirectory)

    for fileName in os.listdir(labelDirectory):
        originalPathToFile = os.path.join(labelDirectory, fileName)

        if int(fileName.split('.')[0]) < testFileMax:
            newPathToFile = os.path.join(labelValidationDirectory, fileName)
            testCount += 1
        else:
            newPathToFile = os.path.join(labelTrainDirectory, fileName)
            trainCount += 1

        os.rename(originalPathToFile, newPathToFile)

    os.rmdir(labelDirectory)
    print(f"[DEBUG] {trainCount=} {testCount=}\n")
