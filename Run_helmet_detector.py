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

#Create directory if it doesnt already exists.
#This is where our helmet detected images will be saved.
if not os.path.exists('images/Helmet_detections'):
    os.mkdir('images/Helmet_detections')

#Get Image Paths in the form of a list
image_paths = list(paths.list_images("images"))
#Looping through every individual path

cam = cv2.VideoCapture(0)

# Load yolo model for cyclist
model, classes, colors, output_layers = cyclistCounter.load_yolo()

while True:
    check, image = cam.read()
    image_path = '/home/bwallima/PycharmProjects/Cyclists-Helmet-Detection/temp_image.jpg'
    cv2.imwrite(image_path, image)

    #Reading the Image from the Path and Resizing it
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    #Just some space to make things look clean
    print('\n')
    #Feeding our individual image to our Helmet Detection Class
    detected_img = cyclistCounter.image_detect(image, cnt_cyclist, model, classes, colors, output_layers)
    frame, outs = helmet_detection.get_detection(frame=image,image_path=image_path, copy_frame=image, numb_cyclist = detected_img, cnt_helmets=cnt_helmets, cnt_no_helmets=cnt_no_helmets)
    
    """
    Press Esc Key to go through the processed images one by one in the Window.
    """
#Print for Fun
print("Total Cyclists wearing Helmets are {}".format(sum(cnt_helmets)))
print("Total Cyclists without Helmets are {}".format(sum(cnt_no_helmets)))
