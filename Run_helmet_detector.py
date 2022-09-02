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

while True:
    start = time.time()
    check, image = cam.read()

    image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    height, width, channels = image.shape

    # Feeding our individual image to our Helmet Detection Class
    numb_cyclist = cyclistCounter.image_detect(image, cnt_cyclist, height, width, model, classes, colors,
                                                        output_layers)
    if numb_cyclist == 0:
        print(time.time())
        continue
    helmet_detection.get_detection(frame=image, copy_frame=image, numb_cyclist=numb_cyclist,
                                   cnt_helmets=cnt_helmets, cnt_no_helmets=cnt_no_helmets, net=net)

    print(f"Loop time: {time.time() - start}")
