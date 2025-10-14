import os
import time
import cv2 as cv
import numpy as np

# Rutas
dataset = 'C:/python_projects/face_recog/rostros/datasets'
models_dir = 'C:/python_projects/face_recog/rostros/trained_models'
os.makedirs(models_dir, exist_ok=True)

faces  = os.listdir(dataset)
print(faces)

labels = []
facesData = []
label = 0 
for face in faces:
    facePath = dataset+'/'+face
    for faceName in os.listdir(facePath):
        img = cv.imread(facePath+'/'+faceName, 0)
        if img is None:
            continue
        img = cv.resize(img, (48, 48), interpolation=cv.INTER_AREA)
        facesData.append(img)
        labels.append(label)
    label = label + 1
#print(np.count_nonzero(np.array(labels)==0)) 
startTime = time.perf_counter()

faceRecognizer = cv.face.FisherFaceRecognizer_create() # type: ignore
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write(os.path.join(models_dir, 'FisherFace2.xml'))

print(f"El modelo tardo {time.perf_counter() - startTime} segundos")
