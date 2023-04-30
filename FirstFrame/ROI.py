#-*- coding: utf-8 -*-
import cv2
import os
import TPE
import Gabor

#face
def rect1(detector,predictor,i,shotname,picturepath,MouthPath,GaborPath,SheetPath,FeaturesPath):

 def Findt1(leftpoint, range1):
        for a in range(0, len(range1)):

            value = range1[a] - leftpoint
            if value == 0:
                return range1[a]
            if value > 0:
                t1 = range1[a - 1]
                # print ('t1=',t1)
                return t1

 def Findt2(rightpoint, range2):
        for a in range(0, len(range2)):

            value = range2[a] - rightpoint
            if value >= 0:
                # print (range2[a])
                return range2[a]

 def Findt3(toppoint, range3):
        for a in range(0, len(range3)):
            # print ('a3=', a)
            # print ('range3[a]', range3[a])
            value = range3[a] - toppoint
            if value == 0:
                # print ('range3[a]=',range3[a])
                return range3[a]
            if value > 0:
                # print (range3[a - 1])
                return range3[a - 1]

 def Findt4(buttompoint, range4):
        for a in range(0, len(range4)):
            value = range4[a] - buttompoint
            if value >= 0:
                # print (range4[a])
                return range4[a]

 img = cv2.imread(picturepath)
 dets = detector(img, 1)

 # create a file PATH to store mouth picture

 if not os.path.exists(MouthPath):
    os.mkdir(os.path.join(MouthPath))
 cur_dir =os.path.join(MouthPath, shotname)
 if not os.path.exists(cur_dir):
     os.mkdir(os.path.join(cur_dir))
 # else:
 #     print 'file exist'


    #face
 for k, d in enumerate(dets):
     shape = predictor(img, d)

     t1 = shape.part(48).x - 10
     t2 = shape.part(54).x + 10
     t3 = shape.part(50).y - 5
     t4 = shape.part(58).y + 5

     mouth_centroid_x = (shape.part(48).x - t1) + abs(shape.part(54).x - shape.part(48).x) / 2
     mouth_centroid_y = (shape.part(51).y - t4) + abs(shape.part(62).y - shape.part(51).y) + abs(
         shape.part(66).y - shape.part(62).y) / 2

     # print("ROI image=",t1,t2,t3,t4)
     ROI_mouth = img[t3:t4, t1:t2]

     widthG = shape.part(64).x - shape.part(60).x
     heightG = shape.part(62).y - shape.part(66).y
     cur_dir = cur_dir + '/'  # 保存路径
     # print(path)

     if not os.path.exists(cur_dir):
         os.mkdir(cur_dir)
     ROIpath = cur_dir + str('%02d' % i) + '.jpg'

     cv2.imwrite(ROIpath, ROI_mouth)
     return ROIpath, mouth_centroid_x, mouth_centroid_y, ROI_mouth, widthG, heightG

