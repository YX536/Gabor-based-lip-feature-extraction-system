#Gabor-based lipreading systems
This repository provides the code for lip feature extraction systems described in [Gabor Based Lipreading with a New Audiovisual Mandarin Corpus](https://link.springer.com/chapter/10.1007/978-3-030-39431-8_16) and [Gabor-based Audiovisual Fusion for Mandarin Chinese Speech Recognition](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://eurasip.org/Proceedings/Eusipco/Eusipco2022/pdfs/0000603.pdf). We also offer an updated version of the lip feature extraction system:  ['FirstFrame' Gabor-based lip feature extraction system](#First).

####Gabor features
<div>
  <video src="https://github.com/YX536/Gabor-based-lip-feature-extraction-system/blob/main/bbae1a.mpg" width="45%" controls></video>
  <video src="https://github.com/YX536/Gabor-based-lip-feature-extraction-system/blob/main/Area.mp4" width="45%" controls></video>
</div>

**Contents**

[TOC]

####Environment:
All package versions are recorded in the "Packages.txt" file.

####Gabor-Based Lip feature extraction system
##### 1. Gabor Based Lipreading with a New Audiovisual Mandarin Corpus
- Handcrafted Gabor-based lip feature extraction system
![Handcrafted Gabor-based lip feature extraction system](https://github.com/YX536/Gabor-based-lip-feature-extraction-system/blob/main/Handcrafted.png)
------------
1). Main.py. 
```python linenums=12
#Change path. Please modify the path in the code to your local path.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # path of "shape_predictor_68_face_landmarks.dat"
VideoPath ='D:/Handcrafted/*.mpg'  # video path 
Frame = 'D:/Handcrafted/Picture/'# path to store frames
MouthPath = 'D:/Handcrafted/mouth/'  # path to store mouth region
GaborPath = 'D:/Handcrafted/Gabor/'#path to store Gabor features
SheetPath = 'D:/Handcrafted/Sheet/' # path to store lip features
FeaturesPath = 'D:/Handcrafted/Features/'  # path to store lip features
```
2). Frame.py. Cut frames from one video. (Frames are stored in 'Picture' folder)

3). ROI.py. Choose the region of interest (ROI) using Dlib 68 point. (Mouth pictures are stored in 'Mouth' folder)

4). Gabor.py .Manually adjust Gabor parameters and generate Gabor features. (Gabor features are stored in 'Gabor' folder)

5). Features.py.  Obtian 7 lip features: Width, height, area, intensity, orientation, the x-value of central point and the y value of central point. (Lip features are instored in 'Feature' folder and 'Sheet' folder)
------------
To run the system, execute "Main.py".


##### 2. Gabor-based Audiovisual Fusion for Mandarin Chinese Speech Recognition
- Optimized Gabor-based lip feature extraction system (code is in "Optimized" folder)
![Optimized Gabor-based lip feature extraction system](https://github.com/YX536/Gabor-based-lip-feature-extraction-system/blob/main/Optimized.png)
------------
1). Main.py. 

2). Frame.py. Cut frames from one video. (Frames are stored in 'Picture' folder)

3). ROI.py. Choose the ROI using Dlib 68 point. (Mouth picture are stored in 'Mouth' folder)

4). TPE.py. Find the most suitable Gabor parameters and iterating them for all frames. 
```python
    #You can change the path of test Gabor path to your local path.
    Gabor_Path = './TPEH.jpg'
```

```python
    #You can change the "Search space", 'iteration times' and 'best loss' according to your requirement.
    #Search range of Gabor parameters
    search_spaceH = {
        "Hkernel_size": hyperopt.hp.quniform('Hkernel_size', 5, 20, 1),
        "Hwavelength": hyperopt.hp.quniform('Hwavelength', 5, 20, 1),
        "Hsig": hyperopt.hp.quniform('Hsig', 3, 7, 1),
        "Hgamma": hyperopt.hp.quniform('Hgamma', 0.3, 0.7,0.1)
    }

    while True:
        try:
            trials = hyperopt.Trials()
            Hbest = hyperopt.fmin(
                        fn=HGabor,
                        space=search_spaceH,
                        algo=hyperopt.tpe.suggest,
                        max_evals=150,             #iteration times
                        trials=trials
                    )


            trial_loss = np.asarray(trials.losses(), dtype=float)
            best_loss = min(trial_loss)
            print('best loss: ', best_loss) 
            if best_loss>10:                           #best loss
                continue
```
5). Gabor.py uses best Gabor parameters to generate gabor features. (Gabor features are stored in 'Gabor' folder)

6). Features.py.  Obtian 7 lip features: Width, height, area, intensity, orientation, the x-value of central point and the y value of central point. (Lip features are instored in 'Feature' folder and 'Sheet' folder)
------------
To run the system, execute "Main.py".

First
##### 3. 'FirstFrame' Gabor-based lip feature extraction system 
!['FirstFrame' Gabor-based lip feature extraction system](https://github.com/YX536/Gabor-based-lip-feature-extraction-system/blob/main/FirstFrame.png)

1). Main.py. 

2). Frame.py. Cut frames from one video. (Frames are stored in 'Picture' folder)

3). ROI.py. Choose the ROI using Dlib 68 point. (Mouth picture are stored in 'Mouth' folder)

4). TPE.py. Find the most suitable Gabor parameters and iterating them for first frames. 

5). Gabor.py uses best Gabor parameters to generate gabor features. (Gabor features are stored in 'Gabor' folder)

6). Features.py.  Obtian 7 lip features: Width, height, area, intensity, orientation, the x-value of central point and the y value of central point. (Lip features are instored in 'Feature' folder and 'Sheet' folder)
------------
To run the system, execute "Main.py".

####Citation
If you find this code useful in your research, please consider to cite the following papers:
```
@inproceedings{xu2020gabor,
  title={Gabor based lipreading with a new audiovisual mandarin corpus},
  author={Xu, Yan and Li, Yuexuan and Abel, Andrew},
  booktitle={Advances in Brain Inspired Cognitive Systems: 10th International Conference, BICS 2019, Guangzhou, China, July 13--14, 2019, Proceedings 10},
  pages={169--179},
  year={2020},
  organization={Springer}
}

@inproceedings{xu2022gabor,
  title={Gabor-based Audiovisual Fusion for Mandarin Chinese Speech Recognition},
  author={Xu, Yan and Wang, Hongce and Dong, Zhongping and Li, Yuexuan and Abel, Andrew},
  booktitle={2022 30th European Signal Processing Conference (EUSIPCO)},
  pages={603--607},
  year={2022},
  organization={IEEE}
}
```
####Contact
[Yan Xu](yan.xu[at]xjtlu.edu.cn)
