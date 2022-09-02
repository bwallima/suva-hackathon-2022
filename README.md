# Suva Hackathon 2022 - Bicycle Helmet Traffic Light

This is a demo prepared for the Suva Hackathon 2022. The challenge was how to
help in accident prevention. We decided for a bicycle helmet traffic light that
would warn cyclist in case they were not wearing a helmet.

It consists of a "real-time" object detection process coupled with a traffic
light display.

The object detection process is built from two pre-trained YOLOv3 CNN models.
The first model detects people and bicycles, while the second model detects
helmets. The output of both models is then used to show a green-red-neutral
display.

This code was forked from https://github.com/CuFFaz/Cyclists-Helmet-Detection
which itself seems to be based on
https://github.com/BlcaKHat/yolov3-Helmet-Detection.

# Weights:
This code needs two sets of trained models to run:
- yolo: trained model for person and bicycle detection.
- yolo_helmet: trained model for helmet detection.

Both can be found in the original [Github repo](https://github.com/CuFFaz/Cyclists-Helmet-Detection).
