#Recognition

# USAGE
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    frame = cv2.flip(frame, -1)
    
    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)
        
        # update the list of names
        names.append(name)

    # loop over the recognized faces
    c = 0
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)
        
        if name == "Unknown":
            c += 1
            cv2.imwrite("alert/"+ str(c) +".jpg", frame[top:bottom,left:right])
            email_user = 'raspberrypi.ranald@gmail.com'
            email_password = 'nivethan2000'
            email_send = 'ranaldn@gmail.com'
                
            subject = 'Alert'
                
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = email_send
            msg['Subject'] = subject
                
            body = 'Hi there, unkown personnel detected in the premises'
            msg.attach(MIMEText(body,'plain'))
                
            filename=str(c)+'.jpg'
            attachment  =open('alert/'+filename,'rb')
                
            part = MIMEBase('application','octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition',"attachment; filename= "+filename)
                
            msg.attach(part)
            text = msg.as_string()
            server = smtplib.SMTP('smtp.gmail.com',587)
            server.starttls()
            server.login(email_user,email_password)
                
            server.sendmail(email_user,email_send,text)
            server.quit()
            

    # display the image to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

#Detection:
import numpy as np
import cv2
#import pkg_resources
#haar_xml = pkg_resources.resource_filename('cv2', 'data/haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_1.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
while True:
    ret, img = cap.read()
    #img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()


#Sensor:

import board
import busio
import time
import math
from urllib.request import urlopen
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

i2c = busio.I2C(board.SCL, board.SDA)
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
ads = ADS.ADS1115(i2c)
while True:
    chan0 = AnalogIn(ads, ADS.P0)
    chan1 = AnalogIn(ads, ADS.P1)
    ro0 = 0.8886
    p0 = 10**((math.log((((5.0 - chan0.voltage)/chan0.voltage)/ro0),10) - 1.617)/(-0.4434))
    
    ro1 = 6.43
    p1co2 = 10**((math.log((((5.0 - chan1.voltage)/chan1.voltage)/ro1),10) - 0.7746)/(-0.36664))
    p1co = 10**((math.log((((5.0 - chan1.voltage)/chan1.voltage)/ro1),10) - 0.6433)/(-0.21966))
    
    
    html = urlopen("https://api.thingspeak.com/update?api_key=XD11GH520180BOU9&field1="+str(p0)+"&field2="+str(p1co2)+"&field3="+str(p1co)).read()
    print(chan0.value, p0)
    print(chan1.value, p1co2)
    print(p1co)
    if (p0 > 200):
        email_user = 'raspberrypi.ranald@gmail.com'
        email_password = 'nivethan2000'
        email_send = 'ranaldn@gmail.com'
                
        subject = 'Alert'
                
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = email_send
        msg['Subject'] = subject
                
        body = 'ALERT, High levels of Smoke detected in the vicinity'
        msg.attach(MIMEText(body,'plain'))
                
        #filename='0.jpg'
#         attachment  =open('alert/'+filename,'rb')
                
#         part = MIMEBase('application','octet-stream')
#         part.set_payload((attachment).read())
        #encoders.encode_base64(part)
        #part.add_header('Content-Disposition',"attachment; filename= "+filename)
                
        #msg.attach(part)
        text = msg.as_string()
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        server.login(email_user,email_password)
                
        server.sendmail(email_user,email_send,text)
        server.quit()
        
    if(p1co2 > 100 or p1co > 50):
        email_user = 'raspberrypi.ranald@gmail.com'
        email_password = 'nivethan2000'
        email_send = 'ranaldn@gmail.com'
                
        subject = 'Alert'
                
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = email_send
        msg['Subject'] = subject
                
        body = 'ALERT, High levels of Air pollution detected in the vicinity'
        msg.attach(MIMEText(body,'plain'))
                
        #filename='0.jpg'
        #attachment  =open('alert/'+filename,'rb')
                
        #part = MIMEBase('application','octet-stream')
        #part.set_payload((attachment).read())
        #encoders.encode_base64(part)
        #part.add_header('Content-Disposition',"attachment; filename= "+filename)
                
        #msg.attach(part)
        text = msg.as_string()
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        server.login(email_user,email_password)
                
        server.sendmail(email_user,email_send,text)
        server.quit()
         
    time.sleep(3)
