# https://becominghuman.ai/face-detection-with-opencv-and-deep-learning-from-video-part-2-592e2dee648#:~:text=Imutils%20are%20a%20series%20of,Python%202.7%20and%20Python%203.

import numpy as np
import argparse
import cv2
import imutils
from imutils.video import VideoStream #for video processing
# The VideoStream displays a video from a local stream (for example from a webcam) and allows accessing the streamed video data from Python
import time

#construct the argument and parse the arguments:
parser=argparse.ArgumentParser()

parser.add_argument("-p","--prototxt",required=True,help="path to Caffe 'deploy' prototxt file")

parser.add_argument("-m","--model",required=True,help="path to Caffe pretrained model")

parser.add_argument("-c","--confidence",type=float,default=0.5,help="minimum probablity to filter weak detections")

args=vars(parser.parse_args())

#load our serialized model from disk 
print("[INFO] loading model...")
net=cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

#initialize the videostream and allow camera sensor to warmup
print("[INFO] starting video stream...")
# input a frame of video by using VideoStream
#vs = VideoStream(frame).start()
#src=0 means default camera
## With VideoStream we have defined that source camera will be “0” which is your laptop default camera and after that,
# we have started the camera, and in last we defined that after start is initialized camera will turn 
# on after 2 seconds so that camera sensor can warm up.
vs=VideoStream(src=0).start() #it will start the web came
time.sleep(2.0) #it is to warm up the camera to settle down light,contrast,inference ,etc

#loop over the frames from the video stream i.e it will run until we stop the video/camera
while True:
	#grab the frame from the threaded video stream and resize it to have max width of 720 pixels
	# With vs.read() we catch our videostream from our camera and then resize frame to fixed widht of 720 pixels.
	frame = vs.read()
	frame = imutils.resize(frame, width=720)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()

    #loop over the detections
	for i in range (0,detections.shape[2]): #shape[2] returns blob image
        # extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]
		#gives confidence score for i face

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

        # compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    #show the output frame
	cv2.imshow("Frame",frame)
	# cv2. waitKey() is the function which bind your code with keyboard and any thing you type will be returned by this function. output from waitKey is then logically AND with 0xFF so that last 8 bits can be accessed
	# waitKey(0) function returns -1 when no input is made whatsoever. As soon the event occurs i.e. a Button is pressed it returns a 32-bit integer.
	# Its argument is the time in milliseconds. The function waits for specified milliseconds for any keyboard event. If you press any key in that time, the program continues.
	key=cv2.waitKey(1) & 0xFF

	#if the key 'q' was pressed ,break from the loop
	if key==ord("q"):
		break

#do a bit of cleanup 
#by destroying the frames
cv2.destroyAllWindows()
vs.stop() #stoping frame

