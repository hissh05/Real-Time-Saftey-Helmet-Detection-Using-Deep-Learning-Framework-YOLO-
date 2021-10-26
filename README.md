# PPE Detection using yolo3 and DeepSORT

## Introduction
In Industry, specially manufacturing industry, Personal Protective Equipment (PPE) like helmet (hard-hat), safety-harness, goggles etc play a very important role in ensuring the safety of workers. However, many accidents still occur, due to the negligence of the workers as well as their supervisors. Supervisors can make mistakes due to the fact that such tasks are monotonous and they may not be able to monitor consistently. This project aims to utilize existing CCTV camera infrastructure to assist supervisors to monitor workers effectively by providing them with real time alerts.

## Functioning
* Input is taken from CCTV cameras
* YOLO is used for detecting persons with proper PPE and those without PPE.
* Deep_SORT allocates unique ids to detected persons and tracks them through consecutive frames of the video.
* An alert is raised if a person is found to be without proper PPE for more than some set duration, say 5 seconds.


## Main Page of Project
![img1](https://github.com/hissh05/helmet-detection/blob/main/demo_images/Main_page.png)

After You click Detect Employee.It will start detecting the employee

## Person With helmet
![img2](https://github.com/hissh05/Helmet_detection/blob/main/demo_images/with_helmet.png)
## Person Without Helmet
![img3](https://github.com/hissh05/Helmet_detection/blob/main/demo_images/without_helmet.png)


## Visual and audio alarm system
 1. visual shown in screen 
 2. a commentary is provided wherein person wearing helmet is appreciated and allowed to proceed to work area and person not wearing safety helmet is warned to wear safety 
 helmet before proceeding to work area.

## Training and Collection Of Data
 * The Data Was collected from the google images with person wearing helmet
 * Trained Data in google cloud

## Exe File
 * Run The Exe file in the link given below The project will run fine
 	(https://drive.google.com/file/d/1Gy_WsC-T6H4xPDCogYjfoVaFkmOwLP40/view?usp=sharing)
 * Extract The zip File and run the Exe File(helmetgui.exe)
 * for code it is attached below

## Acknowledgements

* [rekon/keras-yolo2](https://github.com/rekon/keras-yolo2) for training data.
* [experiencor/keras-yolo3](https://github.com/experiencor/keras-yolo3) for YOLO v3 implementation.
* [nwojke/deep_sort](https://github.com/nwojke/deep_sort) for Deep_SORT implementation.
