#Importing necessary Libraries and Packages.
from main import cyclistCounter, helmet_detection
from imutils import paths
import cv2
import os

###Initializing or helmet detection and counter
#Instantiating our helmet class
helmet_detection = helmet_detection()

#Empty lists for multiple counts.
cnt_cyclist = []
cnt_helmets = []
cnt_no_helmets = []

cam = cv2.VideoCapture(0)

while True:
    check, image = cam.read()
    image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    height, width, channels = image.shape

    print('\n')
    #Feeding our individual image to our Helmet Detection Class
    detected_img = cyclistCounter.image_detect(image, cnt_cyclist, height, width)
    frame, outs = helmet_detection.get_detection(frame=image, copy_frame=image, numb_cyclist = detected_img,
                                                 cnt_helmets=cnt_helmets, cnt_no_helmets=cnt_no_helmets)
    
    """
    Press Esc Key to go through the processed images one by one in the Window.
    """
#Print for Fun
print("Total Cyclists wearing Helmets are {}".format(sum(cnt_helmets)))
print("Total Cyclists without Helmets are {}".format(sum(cnt_no_helmets)))
