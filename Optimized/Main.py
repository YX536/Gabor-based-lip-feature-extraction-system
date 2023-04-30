import glob
import cv2
import os
import WordVideo
import dlib
import time
import ROI
import TPE
import Gabor
import Features
a=time.time()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Video_path = r'D:/FeatureExtractionAuto/H/*.mpg'
VideoPath ='D:/FirstFrame/bbae1a.mpg'
Frame = 'D:/FE/Picture/'# path to store pictures
MouthPath = 'D:/FE/mouth/'  # path to store mouth
GaborPath = 'D:/FE/Gabor/'#path to store Gabor features
SheetPath = 'D:/FE/Sheet/' # path to storSheetPath
FeaturesPath = 'D:/FE/Features/'  # path to store sheets


WordVideo.Frame(detector,predictor,VideoPath,Frame,MouthPath,GaborPath,SheetPath,FeaturesPath)


b=time.time()
print("Time = ",b-a)