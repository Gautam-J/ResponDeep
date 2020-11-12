import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model

from utils import getPathToLatestModel

pathToModel = getPathToLatestModel()
model = load_model(pathToModel)

labelIndices = {
    0: 'background',
    1: 'greet',
    2: 'thumbs_down',
    3: 'thumbs_up',
}

videoCapture = cv2.VideoCapture(0)
while True:
    # TODO: function to count the frames per second in this loop
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

    cv2.putText(frame, f'{classLabel} ({classConfidence:.2f}%)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('ResponDeep', frame)

    time.sleep(1 / 24.)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
