import os
import cv2
import time
import argparse
from utils import getNumberOfPNGInDirectory


def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--label', type=str, default='background',
                        help='Label of the data to be saved. Defaults to "background"')

    args = parser.parse_args()

    return args


args = getArgs()
labelDirectory = os.path.join('data', args.label)
videoCapture = cv2.VideoCapture(0)

if not os.path.exists(labelDirectory):
    os.makedirs(labelDirectory)
    print(f'[INFO] New directory {labelDirectory} created')
else:
    print(f'[INFO] {labelDirectory} exists')

count = getNumberOfPNGInDirectory(labelDirectory)

print('[INFO] Capturing data in 3 secs...')
time.sleep(3)

while True:
    _, frame = videoCapture.read()
    frame = cv2.flip(frame, 1)

    fileName = str(count) + '.png'
    cv2.imwrite(os.path.join(labelDirectory, fileName), frame)
    count += 1

    if (count + 1) % 50 == 0:
        print('Number of images:', count + 1)

    cv2.imshow('ResponDeep', frame)
    time.sleep(1 / 24.)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
