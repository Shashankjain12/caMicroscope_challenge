
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import time
import os
from flask import Flask,render_template,send_file,request
from PIL import Image
import io

app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_img",methods=['GET','POST'])
def upload_img1():
    #filestr = request.files['photos'].read()
    #npimg = np.fromstring(filestr, np.uint8)
    #img = cv2.imdecode(npimg, cv2.CV_LOAD_IMAGE_UNCHANGED)
    img=cv2.imdecode(np.fromstring(request.files['photos'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    cat=cat_detect(img)
    cat=np.array(cat)
    cat_arr=Image.fromarray(cat)
    output = io.BytesIO()
    cat_arr.convert('RGBA').save(output, format='PNG')
    output.seek(0, 0)
    return send_file(output, mimetype='image/png', as_attachment=False)


def cat_detect(frame):
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(["weights", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    config_file="./weights/yolov3.cfg"

    weights_path="./weights/yolov3.weights"
    net=cv2.dnn.readNetFromDarknet(config_file,weights_path)
    (H, W) = frame.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    counts=[]
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            if classID == 15:
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    counts.append(1)
                    
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
        
        # ensure at least one detection exists
        
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                    confidences[i])
                cv2.putText(frame,text,(x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.3)
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i],len(counts))
                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(frame,"No Cat Detected",(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)

        return frame

if __name__=="__main__":
    app.run()
