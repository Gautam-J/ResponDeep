import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils import getPathToLatestModel

labelIndices = {
    0: 'background',
    1: 'greet',
    2: 'thumbs_down',
    3: 'thumbs_up',
}

pathToModel = getPathToLatestModel()
print('[INFO] Path to latest model:', pathToModel)

threshold = 0.7
model = load_model(pathToModel)
videoCapture = cv2.VideoCapture(0)

while True:
    _, frame = videoCapture.read()
    frame = cv2.flip(frame, 1)

    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((1, 224, 224, 3))
    image = preprocess_input(image)

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
