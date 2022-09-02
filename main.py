import cv2
import numpy as np
import time


# Class to Detect Cyclists and Cycles in images
class cyclistCounter():
    # Loading yolo
    def load_yolo():
        net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
        classes = []
        with open("yolo/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layers_names = net.getLayerNames()
        output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        return net, classes, colors, output_layers

    # Detecting out two classes, cyclists and their cycles using YOLOv3
    def detect_objects(img, net, outputLayers):
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(outputLayers)
        return blob, outputs

    # Extracting dimensions of detected objects
    def get_box_dimensions(outputs, height, width):
        boxes = []
        confs = []
        class_ids = []
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                # ignoring all classes except the first two (person & bicycle) of the coco.names list
                if class_id <= 1:
                    if conf > 0.3:
                        center_x = int(detect[0] * width)
                        center_y = int(detect[1] * height)
                        w = int(detect[2] * width)
                        h = int(detect[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confs.append(float(conf))
                        class_ids.append(class_id)
        return boxes, confs, class_ids

    # Drawing Rectangles, Printing Texts, Counting the Total Number of Objects etc
    def draw_labels(boxes, confs, colors, class_ids, classes, img, cnt_cyclist):
        winName = "Cyclist Counter"
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        numb_cyclist = 0
        numb_cycles = 0
        imgh, imgw, _ = img.shape
        print_w = round(5 * imgw / 100)
        print_h = round(5 * imgh / 100)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]

                label = str(classes[class_ids[i]])
                color = colors[i]
                if label == "person":
                    numb_cyclist = numb_cyclist + 1
                elif label == "bicycle":
                    numb_cycles = numb_cycles + 1
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        cnt_cyclist.append(numb_cyclist)
        return numb_cyclist

    # Flow
    @staticmethod
    def image_detect(image, cnt_cyclist, height, width, model, classes, colors, output_layers):
        blob, outputs = cyclistCounter.detect_objects(image, model, output_layers)
        boxes, confs, class_ids = cyclistCounter.get_box_dimensions(outputs, height, width)
        numb_cyclist = cyclistCounter.draw_labels(boxes, confs, colors, class_ids, classes, image, cnt_cyclist)
        return numb_cyclist


# Class to detect Helmets and Non Helmets in images.
class helmet_detection(cyclistCounter):
    # Constructor
    def __init__(self):
        # Initialize the parameters
        self.confThreshold = 0.4  # Confidence threshold
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold
        self.inpWidth = 416  # Width of network's input image
        self.inpHeight = 416  # Height of network's input image
        # Load names of classes
        classesFile = "yolo_helmet/obj.names";
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    # Get the names of the output layers
    @staticmethod
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(self, frame, confidences, indices, count_person, boxes, numb_cyclist, cnt_helmets, cnt_no_helmets):
        winName = "Helmet_Detection"
        imgh, imgw, _ = frame.shape
        fold_path = "images/Helmet_detections"
        print_w = round(8 * imgw / 100)
        print_h = round(8 * imgh / 100)
        for i in indices:
            i = i[0]
            conf = confidences[i]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            right = left + width
            bottom = top + height

            # Draw a bounding box around the object demensions, Print Texts on images, Count several detections to be printed in terminal.
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 3)
            cv2.putText(frame, "helmet", (left, bottom + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        count_no_helmets = numb_cyclist - count_person
        cnt_helmets.append(count_person)
        cnt_no_helmets.append(count_no_helmets)
        cv2.rectangle(frame, (print_w - 30, print_h - 30), (print_w + 320, print_h + 30), (255, 255, 255), -1)
        cv2.putText(frame, "Count of Cyclists with Helmets :{}".format(count_person), (print_w, print_h),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(frame, "Count of Cyclists without Helmets :{}".format(count_no_helmets), (print_w, print_h + 15),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        cv2.imshow(winName, frame)
        cv2.waitKey(27)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs, numb_cyclist, cnt_helmets, cnt_no_helmets):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        count_person = len(indices)  # for counting the classes in this loop.

        # this function in  loop is calling drawPred so, try pushing one test counter in parameter , so it can calculate it.
        self.drawPred(frame, confidences, indices, count_person, boxes, numb_cyclist, cnt_helmets, cnt_no_helmets)
        # increase test counter till the loop end then print...

    # Flow
    def get_detection(self, frame, numb_cyclist, cnt_helmets, cnt_no_helmets, net, copy_frame=None):
        if copy_frame is None:
            copy_frame = frame

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (256, 256), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(self.getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        self.postprocess(copy_frame, outs, numb_cyclist, cnt_helmets, cnt_no_helmets)

        # Put efficiency information.
        # The function getPerfProfile returns the overall time for inference(t) and
        # the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        return copy_frame, outs, numb_cyclist, cnt_helmets, cnt_no_helmets
