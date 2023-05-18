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

from StateMachine import StateMachine
from options import OPTIONS, START_STATE
import time

# sysv_ipc is needed to access the shared memory where the camera image is present.
import sysv_ipc

# numpy and cv2 are needed to access, modify, or display the pixels
import numpy as np

# OD4Session is needed to send and receive messages
import OD4Session

# Import the OpenDLV Standard Message Set.
import opendlv_standard_message_set_v0_9_10_pb2

millis = lambda: round(time.time() * 1000)

START_TIME = millis()

################################################################################

# Create a session to send and receive messages from a running OD4Session;
session = OD4Session.OD4Session(cid=OPTIONS.cid)

# Our state machine
stateMachine = StateMachine(START_STATE, session)

# Register a handler for a message; the following example is listening
# for messageID 1039 which represents opendlv.proxy.DistanceReading.
# Cf. here: https://github.com/chalmers-revere/opendlv.standard-message-set/blob/master/opendlv.odvd#L113-L115
MESSAGE_ID_DISTANCE_READING = 1039
session.registerMessageCallback(
    MESSAGE_ID_DISTANCE_READING,
    lambda msg, senderStamp, timeStamps: stateMachine.onDistance(
        senderStamp, msg.distance
    ),
    opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_DistanceReading,
)
# Connect to the network session.
session.connect()


################################################################################
# The following lines connect to the camera frame that resides in shared memory.
# This name must match with the name used in the h264-decoder-viewer.yml file.
CAMERA_NAME = OPTIONS.cameraName
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
    curr = millis()
    stateMachine.incTime(curr - START_TIME)
    START_TIME = curr

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
    img = np.frombuffer(buf, np.uint8).reshape(
        OPTIONS.height, OPTIONS.width, OPTIONS.channels
    )

    # Run the state
    stateMachine.runState(img)
