import cv2
import os
from datetime import datetime

#Creamos una carpeta para almacenar los rostros si esta no existe
nombre_usuario = input("Introduce el nombre del usuario a registrar:  ")

if not os.path.exists('dataset'):
    os.makedirs('dataset')
    print('Carpeta creada: ', 'dataset')
    
#Por cada usuario nuevo se crea una carpeta    
if not os.path.exists(nombre_usuario):
    ruta = os.path.join('dataset', nombre_usuario)
    os.makedirs(ruta)
    print('Carpeta creada: ', nombre_usuario)
    
#Capturamos vídeo por webcam    
cap = cv2.VideoCapture(0)

#Clasificador haarcascade para identificar rostros
faceClassif = cv2.CascadeClassifier('modelos/haarcascade_frontalface_default.xml')

numcapturas = int(input("Introduce el número de capturas a realizar:  "))

#Extracción de rostros
count = 0
while count < numcapturas:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    #Transformamos imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    
    #Detectamos los rostros en escala de grises
    faces = faceClassif.detectMultiScale(gray, 1.2, 9)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    #Recorte y almacenamiento de rostros
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
        #Recortamos rostro de la imagen de la entrada 
        rostro = auxFrame[y:y+h,x:x+w]
        #Redimensionamos el rostro
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
      
        #Capturamos fecha y hora para el nombre del archivo
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        nombre_archivo = f"imagen_{count}{timestamp}.jpg"
        
        #Almacenamos captura de rostro junto con el nombre de archivo basado en fecha y hora.
        cv2.imwrite(os.path.join(ruta, nombre_archivo), rostro)
        cv2.imshow('rostro',rostro)
        count = count +1
        
        #Mostramos indicador de número de capturas
        cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
        cv2.putText(frame,str(count)+'/'+str(numcapturas),(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)
        cv2.imshow('frame',frame)
            
cap.release()
cv2.destroyAllWindows()
