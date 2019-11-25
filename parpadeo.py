from scipy.spatial import distance  
from imutils.video import VideoStream
from imutils import face_utils
import numpy
import argparse
import imutils
import time
import dlib
import cv2
import cv2 as cv

def eyeDistance(eye):
	#Determinar el valor absoluto de la resta de los primeros 4 
	#puntos de la formula
	p15=distance.euclidean(eye[1],eye[5])
	p24=distance.euclidean(eye[2],eye[4])
	
	#Determinar el valor absoluto de la resta del denominador de la 
	#formula
	
	p03=distance.euclidean(eye[0],eye[3])

	#Resultado de la formula
	ear=(p15 + p24)/(2.0*p03)

	# devuelve el aspecto del ojo (ear)
	return ear

#Parsea los argumentos
argumentParse =	argparse.ArgumentParser()

argumentParse.add_argument("-p","--shape-predictor",help="path to facial landmark predictor")
args=vars(argumentParse.parse_args())
#shape_predictor=args.shape_predictor
#Se define la primera constante para tener un ear como indice de medida
#Se define la segunda contante para determinar cuantos frames pueden permaneces los ojos cerrados

eyesAspect=0.3
consecutiveFrames=3

contador=0
microsueno=0
#Se inicializa el detector de rostros de la libreria dlib
print("inicializando detector de rostos de dlib")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#Se obtienen cuales  son las coordenadas en las cuales se encuentran los ojos.

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]	

#Se comienza la captura de video
cap= cv.VideoCapture(0)
if not cap.isOpened():
	print("Cannot open camera")
	exit()
while True:
	#Capture frame-by-frame
	ret, frame = cap.read()
	#if frame is read correctly ret is true
	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		break
	#our operations on the frame come here
	frame = imutils.resize(frame, width=450)
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	gray = cv.equalizeHist(gray)
	#Display the resulting frame
	cv.imshow('Frame',gray)
	if cv.waitKey(1) == ord('q'):
		break

	#Detecta los rostros
	rects = detector(gray,0)
	#Ciclo sobre la deteccion de rostros
	for rect in rects:
		#determinar los landmarks en la region del rostro, luego convertirlo  a un Numpy array
		shape = predictor(gray,rect)
		shape = face_utils.shape_to_np(shape)

		#obtener las coordenadas de los ojos
	
		leftEye=shape[lStart:lEnd]
		rigthEye=shape[rStart:rEnd]
		leftEar=eyeDistance(leftEye)
		rigthEar=eyeDistance(rigthEye)
	
		#Realizar un promedio del aspecto de los ojos
		ear= (leftEar+rigthEar)/2.0
	
		leftEyeHull= cv2.convexHull(leftEye)
		rigthEyeHull= cv2.convexHull(rigthEye)
		cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
		cv2.drawContours(frame,[rigthEyeHull],-1,(0,255,0),1)

		#verificar si el ear supera el valor que que se puso de referencia
		if ear < eyesAspect:
			contador += 1

		else:
		
			if contador >= consecutiveFrames:
				microsueno+=1
				print("microsueno")

				contador =0
	
#When everything donne, release the capture
cap.release()
cv.destroyAllWindows()







	
