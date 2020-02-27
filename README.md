<h1 align="center"> caMicroscope_code_challenge </h1>
This repository is an approach to solve the challenge of Cancer Region Of Interest extraction from the image which deals with detection of Cancer from Specific area.

Being a coding challenge the main aim is to create a solution for creation of ROI after detection of seperate object in an image. 
As a part of this coding challenge I have used YOLO object detection to detect certain object such as **Cat** in the given image and after detection of the cat creation of certain bounding box where actually cat is there this is done by YOLO algorithm which generally means You Look Only Once in this algorithm only once an images is transmitted to the model and model after creation many bounding box around certain object generates the most appropriate results which specify the actual position with max bounding box coverages to determine the actual position of the object.

Before Object detection,

<img src="./images/cat2.jpeg"></img>

After Object Detection,

<img src="./images/cat_detect.jpeg"></img>


