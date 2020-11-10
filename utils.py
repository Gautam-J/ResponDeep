import os


def getNumberOfPNGInDirectory(directory):
    return len([i for i in os.listdir(directory) if i.endswith('.png')])
