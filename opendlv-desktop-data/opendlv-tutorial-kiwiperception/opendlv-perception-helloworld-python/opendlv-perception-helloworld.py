#!/usr/bin/env python3

# Copyright (C) 2018 Christian Berger
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# sysv_ipc is needed to access the shared memory where the camera image is present.
import sysv_ipc
# numpy and cv2 are needed to access, modify, or display the pixels
import numpy
import numpy as np
import cv2
# OD4Session is needed to send and receive messages
import OD4Session
# Import the OpenDLV Standard Message Set.
import opendlv_standard_message_set_v0_9_10_pb2

################################################################################
# This dictionary contains all distance values to be filled by function onDistance(...).
distances = { "front": 0.0, "left": 0.0, "right": 0.0, "rear": 0.0 };

################################################################################
# This callback is triggered whenever there is a new distance reading coming in.
def onDistance(msg, senderStamp, timeStamps):
    print ("Received distance; senderStamp= %s" % (str(senderStamp)))
    print ("sent: %s, received: %s, sample time stamps: %s" % (str(timeStamps[0]), str(timeStamps[1]), str(timeStamps[2])))
    print ("%s" % (msg))
    if senderStamp == 0:
        distances["front"] = msg.distance
    if senderStamp == 1:
        distances["left"] = msg.distance
    if senderStamp == 2:
        distances["rear"] = msg.distance
    if senderStamp == 3:
        distances["right"] = msg.distance


# Create a session to send and receive messages from a running OD4Session;
# Replay mode: CID = 253
# Live mode: CID = 112
# TODO: Change to CID 112 when this program is used on Kiwi.
session = OD4Session.OD4Session(cid=111)
# Register a handler for a message; the following example is listening
# for messageID 1039 which represents opendlv.proxy.DistanceReading.
# Cf. here: https://github.com/chalmers-revere/opendlv.standard-message-set/blob/master/opendlv.odvd#L113-L115
messageIDDistanceReading = 1039
session.registerMessageCallback(messageIDDistanceReading, onDistance, opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_DistanceReading)
# Connect to the network session.
session.connect()

################################################################################
# The following lines connect to the camera frame that resides in shared memory.
# This name must match with the name used in the h264-decoder-viewer.yml file.
name = "/tmp/img.argb"
# Obtain the keys for the shared memory and semaphores.
keySharedMemory = sysv_ipc.ftok(name, 1, True)
keySemMutex = sysv_ipc.ftok(name, 2, True)
keySemCondition = sysv_ipc.ftok(name, 3, True)
# Instantiate the SharedMemory and Semaphore objects.
shm = sysv_ipc.SharedMemory(keySharedMemory)
mutex = sysv_ipc.Semaphore(keySemCondition)
cond = sysv_ipc.Semaphore(keySemCondition)

################################################################################
# Main loop to process the next image frame coming in.
while True:
    # Wait for next notification.
    cond.Z()
    print ("Received new frame.")

    # Lock access to shared memory.
    mutex.acquire()
    # Attach to shared memory.
    shm.attach()
    # Read shared memory into own buffer.
    buf = shm.read()
    # Detach to shared memory.
    shm.detach()
    # Unlock access to shared memory.
    mutex.release()

    # Turn buf into img array (1280 * 720 * 4 bytes (ARGB)) to be used with OpenCV.
    HEIGHT = 720
    WIDTH = 1280
    img = numpy.frombuffer(buf, numpy.uint8).reshape(HEIGHT, WIDTH, 4)

    ############################################################################
    # TODO: Add some image processing logic here.


    # TODO: Do something with the frame.
    hsv = cv2.cvtColor ( img , cv2.COLOR_BGR2HSV )
    
    # Note : H [0 ,180] , S [0 ,255] , V [0 , 255]
    # Blue
    blueHsvLow = (110 , 50 , 50)
    blueHsvHi = (130 , 255 , 255)
    blueCones =     cv2.inRange ( hsv , blueHsvLow , blueHsvHi )

    # Yellow
    yellowHsvLow2 = (10 , 50 , 50) 
    yellowHsvHi2 = (40 , 255 , 255)
    yellowCones  = cv2.inRange ( hsv , yellowHsvLow2 , yellowHsvHi2)

    # # Combine show
    blueCones = yellowCones
    cv2.imshow("blue", blueCones)
    # # cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
    # cv2.rectangle(img, (0, 0), (WIDTH, HEIGHT / 2), (0,255,0))
    # cv2.rectangle(blueCones, (0, 0), (WIDTH, HEIGHT / 2), (0,255,0))

    # Dilate/Erode
    kernel = np.ones((4, 4), np.uint8)
    dilate = cv2.dilate ( blueCones , kernel , iterations=3)
    cv2.imshow("Dilate", dilate)

    # erode = cv2.erode ( dilate , erode , cv:: Mat () , cv :: Point ( -1 , -1) , iterations , 1 , 1)
    # cv2.imshow ( " Erode " , erode )

    # Canny edge detection
    canny = cv2.Canny ( dilate, 30 , 90 , 3)
    # cv2.imshow ( " Canny edges " , canny )

    # RETR_EXTERNAL 
    # CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow("Counturs" , canny )
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.07 * peri, True)
        # if len(approx) < 10: # Number of corners?
        if peri > 100 and peri < 500:
            [a, b, w, h] = cv2.boundingRect(contour)
            cv2.rectangle(img, (a,b), (a+w,b+h), (0, 0, 255))
            # cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
    #     approx = cv2.approxPolyDP()
    #     cv2.rectangle(img, (50, 50), (100, 100), (0,0,255), 2)
    
    # Invert colors
    # img = cv2.bitwise_not(img)

    # Draw a red rectangle
    # cv2.rectangle(img, (50, 50), (100, 100), (0,0,255), 2)

    # TODO: Disable the following two lines before running on Kiwi:
    cv2.imshow("image", img)
    cv2.waitKey(2)

    ############################################################################
    # Example: Accessing the distance readings.
    print ("Front = %s" % (str(distances["front"])))
    print ("Left = %s" % (str(distances["left"])))
    print ("Right = %s" % (str(distances["right"])))
    print ("Rear = %s" % (str(distances["rear"])))

    ############################################################################
    # Example for creating and sending a message to other microservices; can
    # be removed when not needed.
    angleReading = opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_AngleReading()
    angleReading.angle = 123.45

    # 1038 is the message ID for opendlv.proxy.AngleReading
    session.send(1038, angleReading.SerializeToString());

    ############################################################################
    # Steering and acceleration/decelration.
    #
    # Uncomment the following lines to steer; range: +38deg (left) .. -38deg (right).
    # Value groundSteeringRequest.groundSteering must be given in radians (DEG/180. * PI).
    #groundSteeringRequest = opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_GroundSteeringRequest()
    #groundSteeringRequest.groundSteering = 0
    #session.send(1090, groundSteeringRequest.SerializeToString());

    # Uncomment the following lines to accelerate/decelerate; range: +0.25 (forward) .. -1.0 (backwards).
    # Be careful!
    #pedalPositionRequest = opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_PedalPositionRequest()
    #pedalPositionRequest.position = 0
    #session.send(1086, pedalPositionRequest.SerializeToString());

