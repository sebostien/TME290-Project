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
import numpy as np
import cv2

# OD4Session is needed to send and receive messages
import OD4Session

# Import the OpenDLV Standard Message Set.
import opendlv_standard_message_set_v0_9_10_pb2

#from utils import Vec2, Region
prev_x = 1280/2 
curr_x = 1280/2 #Starting value for direction
#HEIGHT = 480
#WIDTH = 640

################################################################################
# Options
RUNNING_ON_KIWI = False #This is messy as hell
REC_FROM_KIWI = True

################################################################################
# Constants
KIWI_OPTIONS = {
    "width": 640,
    "height": 480,
    "channels": 3,
    "cid": 140,
    "camera_name": "/tmp/img.bgr",
}
REPLAY_OPTIONS = {
    "width": 1280,
    "height": 720,
    "channels": 4,
    "cid": 111,
    "camera_name": "/tmp/img.argb",
}
REC_OPTIONS = {
    "width": 640,
    "height": 480,
    "channels": 4,
    "cid": 111,
    "camera_name": "/tmp/img.argb",
}
OPTIONS = KIWI_OPTIONS if RUNNING_ON_KIWI else REPLAY_OPTIONS 
OPTIONS = REC_OPTIONS if REC_FROM_KIWI else REPLAY_OPTIONS
WIDTH : int = OPTIONS["width"]
HEIGHT : int = OPTIONS["height"]
CHANNELS : int = OPTIONS["channels"]
CAMERA_NAME : str = OPTIONS["camera_name"]

# Default target point in middle of image
prev_x = WIDTH // 2


################################################################################
def comply_with_iso(pos: tuple[int , int]) -> tuple:
    return [HEIGHT - pos[0], -pos[0] + WIDTH // 2]


################################################################################
def get_cone_positions(img, outImg, rectColor: tuple[int, int, int]) : #-> list[Region]:
    # Dilate/Erode
    kernel22 = np.ones((2, 2), np.uint8)
    kernel44 = np.ones((4, 4), np.uint8)

    dilate = cv2.dilate(img, kernel44, (-1, -1), iterations=7)
    erode = cv2.erode(dilate, kernel22, (-1, 0), iterations=7)
    cv2.imshow(" Erode ", erode)

    # Canny edge detection
    edges = 30
    threashold1 = 90
    threashold2 = 3
    canny = cv2.Canny(erode, edges, threashold1, threashold2)

    # RETR_EXTERNAL
    # CHAIN_APPROX_SIMPLE
    contours, _hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow("Counturs" , canny )

    #cones: list[Region] = []
    cones = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        if 100 < peri < 500:
            area = cv2.contourArea(contour)
            [x, y, w, h] = cv2.boundingRect(contour)
            cones.append(x + w / 2, y + h / 2, area)
            cv2.rectangle(outImg, (x, y), (x + w, y + h), rectColor)
    return cones


def get_paper_positions(img, outImg, rectColor: tuple[int, int, int]) : # -> Region:
    # Dilate/Erode
    kernel22 = np.ones((2, 2), np.uint8)
    kernel44 = np.ones((4, 4), np.uint8)

    # erode = cv2.erode ( dilate , erode , cv:: Mat () , cv :: Point ( -1 , -1) , iterations , 1 , 1)
    dilate = cv2.dilate (img, kernel44, (-1,-1), iterations=10)
    erode = cv2.erode (dilate, kernel22, (-1,0), iterations=18)
    #blur = cv2.GaussianBlur(dilate, (11,11), 0)
    cv2.imshow ( " Erode " , erode)

    # Canny edge detection
    edges = 30
    threashold1 = 90
    threashold2 = 3
    canny = cv2.Canny(erode, edges, threashold1, threashold2)

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #cv2.imshow("Counturs" , canny )
    paper = {"area":0, "x":WIDTH/2, "y":HEIGHT} #Maybe it will drive in a circle and look for a paper???
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        print(area)
        if area > 3000: #Also a guess, should be tweaked
            [x, y, w, h] = cv2.boundingRect(contour)
            if area > paper["area"] :
                paper = {"area":area, "x":x+w/2, "y":y+h/2}
                cv2.rectangle(outImg, (x,y), (x+w,y+h), rectColor)
    return paper


################################################################################
# This dictionary contains all distance values to be filled by function onDistance(...).
distances = {"front": 0.0, "left": 0.0, "right": 0.0, "rear": 0.0}


################################################################################
# This callback is triggered whenever there is a new distance reading coming in.
def onDistance(msg, senderStamp, timeStamps):
    print("Received distance; senderStamp= %s" % (str(senderStamp)))
    print(
        "sent: %s, received: %s, sample time stamps: %s"
        % (str(timeStamps[0]), str(timeStamps[1]), str(timeStamps[2]))
    )
    print("%s" % (msg))
    if senderStamp == 0:
        distances["front"] = msg.distance
    if senderStamp == 1:
        distances["left"] = msg.distance
    if senderStamp == 2:
        distances["rear"] = msg.distance
    if senderStamp == 3:
        distances["right"] = msg.distance


# Create a session to send and receive messages from a running OD4Session;
# Replay mode: CID = 111
# Live mode: CID = 140
# Change to CID 140 when this program is used on Kiwi.
session = OD4Session.OD4Session(cid=OPTIONS["cid"])
# Register a handler for a message; the following example is listening
# for messageID 1039 which represents opendlv.proxy.DistanceReading.
# Cf. here: https://github.com/chalmers-revere/opendlv.standard-message-set/blob/master/opendlv.odvd#L113-L115
MESSAGE_ID_DISTANCE_READING = 1039
session.registerMessageCallback(
    MESSAGE_ID_DISTANCE_READING,
    onDistance,
    opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_DistanceReading,
)
# Connect to the network session.
session.connect()

################################################################################
# The following lines connect to the camera frame that resides in shared memory.
# This name must match with the name used in the h264-decoder-viewer.yml file.
CAMERA_NAME = OPTIONS["camera_name"]
# Obtain the keys for the shared memory and semaphores.
keySharedMemory = sysv_ipc.ftok(CAMERA_NAME, 1, True)
keySemMutex = sysv_ipc.ftok(CAMERA_NAME, 2, True)
keySemCondition = sysv_ipc.ftok(CAMERA_NAME, 3, True)
# Instantiate the SharedMemory and Semaphore objects.
shm = sysv_ipc.SharedMemory(keySharedMemory)
mutex = sysv_ipc.Semaphore(keySemCondition)
cond = sysv_ipc.Semaphore(keySemCondition)

################################################################################
# Main loop to process the next image frame coming in.
while True:
    # Wait for next notification.
    cond.Z()
    print("Received new frame.")

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
    img = np.frombuffer(buf, np.uint8).reshape(HEIGHT, WIDTH, CHANNELS)

    ############################################################################
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Remove car
    cv2.rectangle(hsv, (512, 600), (768, HEIGHT), (0, 0, 0), -1)
    # Remove top half
    cv2.rectangle(hsv, (0, 0), (WIDTH, HEIGHT // 2), (0, 0, 0), -1)

    # Note : H [0 ,180] , S [0 ,255] , V [0 , 255]
    # Blue paper
    paperHsvLow = (90, 100, 50) #(100, 128, 61), (96, 148, 59) (97, 145, 66)
    paperHsvHi = (110 , 200 , 150)
    bluePaper = cv2.inRange ( hsv , paperHsvLow , paperHsvHi )
    cv2.imshow("bluePaper", bluePaper)
    # Blue
    blueHsvLow = (100, 50, 50)
    blueHsvHi = (130, 255, 255)
    blueCones = cv2.inRange(hsv, blueHsvLow, blueHsvHi)
    # cv2.imshow("Blue Cones", blueCones)

    # Yellow
    yellowHsvLow2 = (15, 50, 50)
    yellowHsvHi2 = (35, 255, 255)
    yellowCones = cv2.inRange(hsv, yellowHsvLow2, yellowHsvHi2)

    #yellow = get_cone_positions(yellowCones, img, (255, 255, 0))
    #blue = get_cone_positions(blueCones, img, (0, 0, 255))
    paper = get_paper_positions(bluePaper, img, (255, 0 , 0))
    print(paper)

    #max_y_blue = WIDTH
    #min_y_yellow = 0
    #for p in blue:
    #    max_y_blue = min(p["x"], max_y_blue)
    #for p in yellow:
    #    min_y_yellow = max(p["x"], min_y_yellow)

    #cv2.rectangle(img, (int(min_y_yellow), 250), (int(max_y_blue), HEIGHT-100), (255, 255, 0))
    #prev_x = int(curr_x)
    #curr_x = (int(min_y_yellow) + int(max_y_blue)) // 2
    #x = prev_x + int(((curr_x - prev_x) * 0.6)) #Some sort of tröghet, can be tuned for sure
    #cv2.rectangle(img, (x, 0), (x, HEIGHT), (0, 255, 255))
    #goal_pos = comply_with_iso({"x": x, "y": 0}) #Cones
    #paperMiddleX = paper["x"] + paper["w"] / 2
    #paperMiddleY = paper["y"] + paper["h"] / 2
    paperx = int(paper["x"])
    papery = int(paper["y"])
    cv2.rectangle(img, (paperx, papery), (paperx + 10, papery + 10), (0, 255, 255), -1)
    goal_pos = comply_with_iso([paperx, papery])

    if not RUNNING_ON_KIWI:
        #cv2.rectangle(img, (x, 0), (x, HEIGHT), (0, 255, 255)) # Yellow aim-line
        cv2.imshow("image", img)
        cv2.waitKey(2)

    ############################################################################
    # Example: Accessing the distance readings.
    print("Front = %s" % (str(distances["front"])))
    print("Left = %s" % (str(distances["left"])))
    print("Right = %s" % (str(distances["right"])))
    print("Rear = %s" % (str(distances["rear"])))

    ############################################################################
    # Example for creating and sending a message to other microservices; can
    # be removed when not needed.
    # angleReading = opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_AngleReading()
    # angleReading.angle = 123.45

    # 1038 is the message ID for opendlv.proxy.AngleReading
    # session.send(1038, angleReading.SerializeToString());

    ############################################################################
    # Steering and acceleration/decelration.
    #
    # Uncomment the following lines to steer; range: +38deg (left) .. -38deg (right).
    # Value groundSteeringRequest.groundSteering must be given in radians (DEG/180. * PI).
    if RUNNING_ON_KIWI:
        groundSteeringRequest = (
            opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_GroundSteeringRequest()
        )
        # In range [-38 deg, 38 deg]
        steer_deg = (goal_pos.y / (WIDTH // 2)) * 38
        steer_rad = steer_deg / 180 * 3.141
        groundSteeringRequest.groundSteering = steer_def
        session.send(1090, groundSteeringRequest.SerializeToString())

        # Uncomment the following lines to accelerate/decelerate; range: +0.25 (forward) .. -1.0 (backwards).
        # Be careful!
        pedalPositionRequest = (
            opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_PedalPositionRequest()
        )
        pedalPositionRequest.position = 0.11
        session.send(1086, pedalPositionRequest.SerializeToString())
