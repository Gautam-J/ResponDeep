import os
import numpy as np

from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report

from model_configs import buildModel, getModelCallbacks
from utils import createBaseDir, getNumberOfClasses
from generators import getTrainGenerator, getValGenerator, getTestGenerator

from visualizations import (plotClassificationReport,
                            plotConfusionMatrix,
                            plotTrainingHistory)

baseDir = createBaseDir()
trainDir = os.path.join('data', 'train')
testDir = os.path.join('data', 'validation')

batchSize = 8
nEpochs = 10
imgHeight = 224
imgWidth = 224

trainGenerator = getTrainGenerator(trainDir,
                                   (imgWidth, imgHeight),
                                   batchSize)

valGenerator = getValGenerator(trainDir,
                               (imgWidth, imgHeight),
                               batchSize)

testGenerator = getTestGenerator(testDir,
                                 (imgWidth, imgHeight),
                                 batchSize)

nClasses = getNumberOfClasses(trainDir)
model = buildModel(input_shape=(imgWidth, imgHeight, 3), n_classes=nClasses)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

plot_model(model, to_file=f'{baseDir}/model_graph.png', show_shapes=True, dpi=200)

stepsPerEpoch = trainGenerator.samples // trainGenerator.batch_size
validationSteps = valGenerator.samples // valGenerator.batch_size
callBacks = getModelCallbacks(baseDir)

history = model.fit(trainGenerator, epochs=nEpochs,
                    steps_per_epoch=stepsPerEpoch,
                    validation_data=valGenerator,
                    validation_steps=validationSteps,
                    callbacks=callBacks)

print('[INFO] Plotting training history')
plotTrainingHistory(history, baseDir)

testLoss, testAccuracy = model.evaluate(testGenerator)
print('[INFO] Test Metrics:')
print(f'[INFO] {testLoss = }')
print(f'[INFO] {testAccuracy = }')

model.save(f'{baseDir}/final_model_{testLoss:.4f}_{testAccuracy:.4f}.h5')

y_pred = np.argmax(model.predict(testGenerator), axis=-1)
y_test = testGenerator.classes
classLabels = list(testGenerator.class_indices.keys())

report = classification_report(y_test, y_pred,
                               target_names=classLabels,
                               output_dict=True)

matrix = confusion_matrix(y_test, y_pred)
matrix = matrix / matrix.astype(np.float).sum(axis=0)  # normalize confusion matrix

print('[INFO] Plotting classification report')
plotClassificationReport(report, baseDir)

print('[INFO] Plotting confusion matrix')
plotConfusionMatrix(matrix, classLabels, baseDir)
