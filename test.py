import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils import getPathToLatestModel, getLabelIndices, loadComicImage

pathToModel = getPathToLatestModel()
model = load_model(pathToModel)
labelIndices = getLabelIndices()

helloImage = loadComicImage(os.path.join('comic_images', 'hello.png'))
yesImage = loadComicImage(os.path.join('comic_images', 'yes.png'))
noImage = loadComicImage(os.path.join('comic_images', 'no.png'))
rows, cols, channels = helloImage.shape

alpha = 0.2
xOffset = (480 - cols) // 2
yOffset = (640 - rows) // 2

videoCapture = cv2.VideoCapture(0)
while True:
    _, frame = videoCapture.read()
    frame = cv2.flip(frame, 1)

    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((1, 224, 224, 3))
    image = (image / 127.0) - 1

    prediction = model.predict(image)[0]

    classIndex = np.argmax(prediction)
    classConfidence = prediction[classIndex] * 100
    classLabel = labelIndices.get(classIndex)

    addedImage = None
    if classLabel == 'greet':
        addedImage = cv2.addWeighted(frame[xOffset:xOffset + rows, yOffset:yOffset + cols, :], alpha, helloImage[0:rows, 0:cols], 1 - alpha, 0)
    elif classLabel == 'thumbs_down':
        addedImage = cv2.addWeighted(frame[xOffset:xOffset + rows, yOffset:yOffset + cols, :], alpha, noImage[0:rows, 0:cols], 1 - alpha, 0)
    elif classLabel == 'thumbs_up':
        addedImage = cv2.addWeighted(frame[xOffset:xOffset + rows, yOffset:yOffset + cols, :], alpha, yesImage[0:rows, 0:cols], 1 - alpha, 0)

    if addedImage is not None:
        frame[xOffset:xOffset + rows, yOffset:yOffset + cols] = addedImage

    cv2.putText(frame, f'{classLabel} ({classConfidence:.2f}%)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('ResponDeep', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
