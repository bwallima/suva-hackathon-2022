from main import cyclistCounter, helmet_detection
import cv2
import time

helmet_detection = helmet_detection()
cnt_cyclist = []
cnt_helmets = []
cnt_no_helmets = []

# load models
model, classes, colors, output_layers = cyclistCounter.load_yolo()
modelConfiguration = "yolo_helmet/yolov3-obj.cfg"
modelWeights = "yolo_helmet/yolov3-obj_2400.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cam = cv2.VideoCapture(0)
trafficLight = trafficlight.TrafficLight()
while True:
    start = time.time()
    check, image = cam.read()

    image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    height, width, channels = image.shape

    # Feeding our individual image to our Helmet Detection Class
    cnt_cyclist = cyclistCounter.image_detect(image, cnt_cyclist, height, width, model, classes, colors,
                                                        output_layers)
    if cnt_cyclist == 0:
        print(time.time())
        continue
    frame, outs,  numb_cyclist, cnt_helmets, cnt_no_helmets = helmet_detection.get_detection(frame=image, copy_frame=image, numb_cyclist=cnt_cyclist,
                                   cnt_helmets=cnt_helmets, cnt_no_helmets=cnt_no_helmets, net=net)

    print(f"Loop time: {time.time() - start}")

    if numb_cyclist > 0:
        if (cnt_helmets[-1] > 0 and cnt_helmets[-1] > cnt_no_helmets[-1]):
            #smile
            trafficLight.update_image(1)
        elif cnt_no_helmets[-1] > 0:
            #cry
            trafficLight.update_image(2)
        else:
            #neutral
            trafficLight.update_image(0)
    else:
        #neutral
        trafficLight.update_image(0)
