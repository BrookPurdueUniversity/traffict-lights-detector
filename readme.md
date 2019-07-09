
# Traffic Lights Detector 
* -- Brook Cheng
* -- July 2 2019

# Introduction

This project is a branch for the [SDCN Capstone Project of Udacity](https://github.com/udacity/CarND-Capstone), in which there are two parts related to the traffic lights: 

* Traffic Lights Detector
* Traffic Lights Classifier

In this part, I have trained a deep learning model to detect the traffic lights in pixel-wise level. That is, I ran semantic segmentation work regarding to driving scenes, with two semantic classes: Traffic Lights and Background. After finishing training the model, we apply it to predict semantic results of other new images with/without traffic lights, the results are compelling.

# Dependencies

This project requires Python 3.7 and the following Python libraries installed:
* [tensorflow-gpu](https://www.tensorflow.org/)
* [keras](https://keras.io/)
* [opencv-python](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)
* [skimage](https://scikit-image.org/)
* [numpy](http://www.numpy.org/)
* [argparse](https://docs.python.org/3/library/argparse.html)
* [glob](https://docs.python.org/3/library/glob.html)

#### Remind: To faciliate traning and testing process, I use a NVIDIA Graphics Card to proceed 
![NVDIA GeForce GTX 780 Ti](//live.staticflickr.com/65535/48235926726_8ca02a533e_h.jpg)


# Dataset

Dataset for training was provided by Udacity from ros bag (traffic_light_bag_files). It's ignored because files can be downloaded from Udacity website and unpack using RosBag instructions.

In this project, we have two different dataset carla and sim, the former is based on the real driving scence and the latter is from the [Autoware](https://github.com/autowarefoundation/autoware) simultor platfrom. Each dataset contains dataset for traning and testing, respectively. Moreover, each pair of images consists of one rgb image and one maks/label. 

For the maski mages, there are 2 semantic classes: Tarffic Lights and Background
![](//live.staticflickr.com/65535/48235925376_0fa2977ebe_b.jpg)

Below are examples of images from these two different datasets.

#### 1. Example from dataset of carla

* image -- ![](//live.staticflickr.com/65535/48235762867_ebc343aa99_c.jpg)
* mask -- ![](//live.staticflickr.com/65535/48235690456_d84238eb00_c.jpg)

#### 2. Example from dataset of simulator

* image -- ![](//live.staticflickr.com/65535/48235705681_85998d770e_c.jpg)
* mask -- ![](//live.staticflickr.com/65535/48235706456_363d0b3c1f_c.jpg)

# File System

./data/
      carla/
           train/
                green/*.jpg
                no/*.jpg
                red/*.jpg
                yellow/*.jpg
           test/
               green/*.jpg
               no/*.jpg
               red/*.jpg
               yellow/*.jpg
       sim/
          train/
               green/*.jpg
               no/*.jpg
               red/*.jpg
               yellow/*.jpg
          test/
              green/*.jpg
              no/*.jpg
              red/*.jpg
              yellow/*.jpg

# Data Preprocessing

* Writing the raw data into .npy file, in which all the images have been saved as numpy.darray format.
* Cropping each image into specific size.
* Transforming RGB images into Gray images.

# Architecture of Neural Networks

For the task of traffic light detection, we adopt [U-Net](https://arxiv.org/abs/1505.04597).

As our input image has been resize and grayed into (W,H,C) = (96,128,1). Thereby, the neural network becomes:

![](//live.staticflickr.com/65535/48235939971_f0e84f8a8f_b.jpg)
![](//live.staticflickr.com/65535/48236019627_469a11b48e_b.jpg)

# Result

I have saved the training weights for dataset of carla and sim, both. 

* tl_weights_detector_carla.h5
* tl_weights_detector_sim.h5

To quick test our model, you can run:

#### python predict.py


```python

```
