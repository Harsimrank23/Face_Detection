#https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
#step 1: face detection
#step 2: face recognization

#open cmd in folder and type: 
#1.pip install opencv-python
##Computer vision is a process by which we can understand the images and videos 
#how they are stored and how we can manipulate and retrieve data from them.
##OpenCV is the huge open-source library for the computer vision, machine learning,
#and image processing

#2.pip install imutils
##Imutils are a series of convenience functions to make basic image processing functions 
#such as translation, rotation, resizing, skeletonization, and displaying Matplotlib 
#images easier with OpenCV

#3.pip install argparse
##it is used for argument passing

#4.pip install pickle
##used for encoding/decoding of detected images

#5.pip install scikit-learn
##machine learning library for training our model

#6.pip install numpy
##for numerical calculations

#import necessary packages:
import numpy as np
import argparse 
import cv2

#construct the argument parse and parse the arguments
#argparse is passing command line argument to the python script
#initializing parser
#we can also pass decription in the ArgumentParser(description:"my face detection project") (optional)
parser=argparse.ArgumentParser()

#adding the parameters u want to pass:
#we can add two types of arguments :positional/optional
#start working with object
#https://zetcode.com/python/argparse/#:~:text=The%20argparse%20module%20makes%20it,defined%20arguments%20from%20the%20sys.&text=A%20parser%20is%20created%20with,optional%2C%20required%2C%20or%20positional.
#The help option gives argument help
#if we run the program with -h keyword then help description will be provided to pass the argument
#here we are working with optional parameters they are given by --name , and during execution we had to pass the name of the arguments in case of optional parameters
#in case of optional paramters position doesnt matter in passing arguments from terminal
#we can pass short form with the the name --> '-shortform'
#obj.add_argument("-shortform","name",required=True,help="",type(to specify type of input),default=(if input in not given default value is taken))
#To make an option required, True can be specified for the required= keyword argument to add_argument():
parser.add_argument("-i","--image",required=True,help="path to input image")

##The prototxt is a text file that holds information about the structure of the neural network: A list of layers in the neural network. The parameters of each layer, such as its name, type, input dimensions,
#and output dimensions. The connections between the layers
##Caffe is a deep learning framework
##The deploy prototxt is basically a duplicate of the train prototxt
##deploy means to create the environment to run the application
parser.add_argument("-p","--prototxt",required=True,help="path to Caffe 'deploy' prototxt file")

parser.add_argument("-m","--model",required=True,help="path to Caffe pretrained model")

parser.add_argument("-c","--confidence",type=float,default=0.5,help="minimum probablity to filter weak detections")

#passing the arguments added:
#vars convert object into dictionary
args=vars(parser.parse_args()) #now all the arguments will be in the args variable in the from of list/dict

#load our serialized model from disk 
print("[INFO] loading model...")
#opencv library has inbuilt function to implement deep neural network(dnn) 
##The latest OpenCV includes a Deep Neural Network (DNN) module, which comes with a nice pre-trained face detection convolutional
#neural network (CNN). The new model enhances the face detection performance compared to the traditional models, such as Haar. The
#framework used to train the new model is Caffe.
net=cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

#reading the inputed image
#load the input image and construct an input blob for the image
#BLOB stands for Binary Large OBject ,blobs can store binary data, they can be used to store images or other multimedia files.
#by resiizing to a fixed 300x300 pixels and then normalizing it
#cv2 had imread to read image from the list of command line arguments
image=cv2.imread(args["image"])

#taking height and width of the image
(h,w)=image.shape[:2] #This is a combination of slicing and sequence unpacking,syntax is start:stop:step.So here it's saying to take the first two elements of the shape attribute.
#shape is (height × width × other dimensions) then you can grab just height and width by taking the first two elements, [:2], and unpacking them appropriately.

#cv2.dnn has blobfromimage method to create a blob of image
#OpenCV’s new deep neural network (dnn ) module contains two functions that can be used for preprocessing images and preparing them for classification via pre-trained deep learning models.
#cv2.dnn.blobFromImage  and cv2.dnn.blobFromImages
#Mean subtraction,Scaling And optionally channel swapping
#blob=cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
#image : This is the input image we want to preprocess before passing it through our dnn for classification.
#scalefactor : After we perform mean subtraction we can optionally scale our images by some factor. This value defaults to `1.0` (i.e., no scaling)
#mean : These are our mean subtraction values. They can be a 3-tuple of the RGB(red,green,blue pixels) means or they can be a single value
#calculate a 300 by 300 pixel blob  from our image 
blob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0))

#pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")

##With setInput() we are setting new input value for the network which is our “blob” and with net.forward() we are scanning the whole image and detecting where the faces are.
#Now we are looping through detections and drawing boxes around detected faces.
net.setInput(blob)
detections=net.forward()

#loop over the detections
for i in range (0,detections.shape[2]): #shape[2] returns blob image
    #extract the confidence (i.e,probability) associated with the prediction
    #confidence is a minimum threshhold value above which we can detect the face and label it [0-1]
    #detections is a 4-D matrix
    confidence=detections[0,0,i,2] #gives confidence score for i face

    #filter out weak detections by ensuring the 'confidence' is greater than minimum confidence
    if confidence>args["confidence"]:
        #compute the (x,y) coordinates of the bounding box for the object
        #detections[0,0,i,3:7] gives the bounding box
        ##The output coordinates of the bounding box are normalized between [0,1]. Thus the coordinates should be multiplied by the height
        #and width of the original image to get the correct bounding box on the image
        box=detections[0,0,i,3:7]*np.array([w,h,w,h]) 
        (startX,startY,endX,endY)=box.astype("int")

        #draw the bounding box of the face along with the associated probability
        ##the % as a special character you need to add, so it can know, that when you type “f”, the number (result) that will be printed will be a 
        #floating point type, and the “.2” tells your “print” to print only the first 2 digits after the point
        text="{:.2f}%".format(confidence*100)
        # In general, we want the label to be displayed above the rectangle, but if there isn’t room, we’ll display it just below the top of the rectangle
        y=startY-10 if startY-10>10 else startY+10
        #rectangle(img, pt1, pt2, color(BGR percentages), thickness)
        cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),2)
        #putText(img, text, org(coordinates), fontFace, fontScale, color, thickness=None)
        #FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, , etc.-->type of font
        # fontScale: Font scale factor that is multiplied by the font-specific base size.
        # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
        cv2.putText(image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)

#show the output image
cv2.imshow("Output",image)
cv2.waitKey(0)
#waitKey(0) will display the window infinitely until any keypress (it is suitable for image display). 2. waitKey(1) will display a frame for 1 ms, after which display will be automatically closed.
