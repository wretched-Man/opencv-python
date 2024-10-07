# Smile Detector
This simple project begins with a discussion on facial landmark detection. This background is what fuels and enables us to create the smile detector.

### Models
Two models are used in this repo. One model is the same face detection model as in the other models and the other is the LBF facial landmarks model. I have not included the model weights in this upload. However, it can be found [here](https://github.com/kurnianggoro/GSOC2017/blob/master/data/lbfmodel.yaml).

### Files
There is one special file in this folder that may need explaining. And that file is the `smile_det_constant_creator.py` file. This file shows how the constants used in the smile detector were found.

## Extensibility
This application only shows the beginnings of what can be done. You could think of more powerful landmark detectors like Google's mediapipe and face-tracking models to create other remarkable applications like blink detectors or even emotion detectors. With face-tracking, even an interests' observation application is possible.
In line with this, the biggest improvement that can be made to this code is using a different landmark detector that does not necessarily depend on front facing faces.
