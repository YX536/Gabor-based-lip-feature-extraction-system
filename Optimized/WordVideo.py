import glob
import os
import ROI
import TPE
import Features
import Gabor
import cv2



def Frame(detector, predictor,VideoPath,Frame,MouthPath,GaborPath,SheetPath,FeaturesPath):

    if not os.path.exists(Frame):
        os.mkdir(Frame)
    #obtain the individual words


    video=glob.glob(VideoPath)
    for m in range(len(video)):
            print(video[m])
    # for video in glob.glob(VideoPath):  # path of videos
            (filepath, tempfilename) = os.path.split(video[m])
            # print(filepath,tempfilename)
            (video_shotname, extension) = os.path.splitext(tempfilename)
            folder_name = video_shotname
            Path = os.path.join(Frame, folder_name)
            print(Path)
            if not os.path.exists(Path):
                os.mkdir(Path)



            # frames = int(clip.fps * clip.duration)

            cap = cv2.VideoCapture(video[m])
            fps = cap.get(5)
            totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # print(fps,totalFrameNumber)
            i = 0
            count=0
            while (cap.isOpened()):  # cv2.VideoCapture.isOpened()
                i = i + 1
                ret, frame = cap.read()  # cv2.VideoCapture.read()　
                if ret == True:
                    path = Path + '/'
                    picturepath = path + str('%02d' % i) + '.jpg'
                    print(picturepath)
                    # print picturepath
                    cv2.imwrite(picturepath, frame)

                    ROIpath, mouth_centroid_x, mouth_centroid_y, ROI_mouth, widthG, heightG = ROI.rect1(
                                        detector, predictor, i, folder_name, picturepath, MouthPath,GaborPath,SheetPath,FeaturesPath)
                    global HGamma, HKernelSize, HSig, HWavelength
                    while True:
                        try:
                            if count == 0:
                                HGamma, HKernelSize, HSig, HWavelength = TPE.TPE(picturepath, mouth_centroid_x,
                                                                                                 mouth_centroid_y, ROI_mouth,
                                                                                                 widthG, heightG)
                            Gabor_Path = Gabor.Gabor_h(HGamma, HKernelSize, HSig, HWavelength, i, ROIpath,
                                                                       folder_name, GaborPath)
                            Features.Features(mouth_centroid_x, mouth_centroid_y, i, folder_name,
                                                              Gabor_Path, SheetPath, FeaturesPath)
                        except Exception as e:
                            print("Try again")
                            continue
                        break
                    count+=1

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
    #
