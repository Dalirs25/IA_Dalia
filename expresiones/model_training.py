import cv2 as cv
import os
import time
import numpy as np

dataset = 'C:/python_projects/expression_recog/images/train'
models_dir = 'C:/python_projects/expression_recog/expresiones/modelo'

expressions = os.listdir(dataset)
print(expressions)

labels = []
expressionsData = []

labelIndex = 0

for expresssion in expressions:
    # Concatenamos el path completo del dataset de la expresion
    expPath = dataset + '/' + expresssion

    for exprName in os.listdir(expPath):
        img = cv.imread(expPath + '/' + exprName, 0)

        if img is None:
            continue
        
        expressionsData.append(img)
        labels.append(labelIndex)

    labelIndex += 1

start_time = time.perf_counter()

faceRecog = cv.face.FisherFaceRecognizer_create()  # type: ignore
faceRecog.train(expressionsData, np.array(labels))
faceRecog.write(os.path.join(models_dir, 'FisherFace.xml'))

print(f"El modelo tardo ({time.perf_counter() - start_time}) segundos en entrenar")

