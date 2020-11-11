import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
from utils import getPathToLatestModel

labelIndices = {
    0: 'background',
    1: 'greet',
    2: 'thumbs_down',
    3: 'thumbs_up',
}

pathToModel = getPathToLatestModel()
print('[INFO] Path to latest model:', pathToModel)

model = load_model(pathToModel)
videoCapture = cv2.VideoCapture(0)

while True:
    _, frame = videoCapture.read()
    frame = cv2.flip(frame, 1)

    image = cv2.resize(frame, (50, 50))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape((1, 50, 50, 1))
    image = image / 255.

    prediction = model.predict(image)[0]
    classLabel = labelIndices.get(np.argmax(prediction))

    cv2.putText(frame, classLabel, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ResponDeep', frame)

    time.sleep(1 / 24.)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
