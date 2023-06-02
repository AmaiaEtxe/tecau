import cv2
import os
import numpy as np
from tqdm import tqdm

#Estructura del directorio:
# dataset
#   |-persona1
#   |    |-imagenes(.jpg)
#   |-persona2
#        |-imagenes(.jpg)

dataPath = 'dataset' #Ruta al dataset
peopleList = os.listdir(dataPath)

# Filtrar solo los directorios de la lista
peopleList = [item for item in peopleList if os.path.isdir(os.path.join(dataPath, item))]

print('Lista de personas: ', peopleList)

#Arrays para etiquetado de imagenes para el entrenamiento
labels = []
facesData = []
label = 0

#Lectura de rostros y etiquetado
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')
    #Filtrado de imagenes de rostros detectados en el directorio 'personPath'
    lista_caras_registradas = [archivo for archivo in os.listdir(personPath) if archivo.endswith('.jpg')]
    
    #Barra de progreso
    with tqdm(total=len(lista_caras_registradas), desc="Progreso de lectura y clasificación de imágenes") as pbar:
        #Recorremos cada una de los rostros del directorio
        for fileName in lista_caras_registradas:
            #Almacenar la etiqueta para la imagen en el array
            labels.append(label)
            #Almacenar la imagen en el array
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
            pbar.update(1)  # Actualizar el progreso de la barra
        label = label + 1
        
# Se selecciona el método LBPH para entrenar el reconocedor
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Se entrena el modelo reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

#Creamos carpeta para guardar el modelo entrenado si no existiera
if not os.path.exists('modelos'):
    os.makedirs('modelos')
    print('Carpeta creada: ', 'modelos')

# Se almacena el modelo obtenido en formato XML
face_recognizer.write('modelos/modeloLBPHFace.xml')
print("Modelo almacenado")
