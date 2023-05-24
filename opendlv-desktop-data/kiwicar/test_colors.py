from StateMachine import StateMachine
import sys
from options import State, OPTIONS
import cv2
import sysv_ipc
import numpy as np

if __name__ == "__main__":
    stateMachine = StateMachine(State.DEBUG_COLORS, None)

    try:
        CAMERA_NAME = OPTIONS.cameraName
        # Obtain the keys for the shared memory and semaphores.
        keySharedMemory = sysv_ipc.ftok(CAMERA_NAME, 1, True)
        keySemMutex = sysv_ipc.ftok(CAMERA_NAME, 2, True)
        keySemCondition = sysv_ipc.ftok(CAMERA_NAME, 3, True)
        # Instantiate the SharedMemory and Semaphore objects.
        shm = sysv_ipc.SharedMemory(keySharedMemory)
        mutex = sysv_ipc.Semaphore(keySemCondition)
        cond = sysv_ipc.Semaphore(keySemCondition)
        mutex.acquire()
        shm.attach()
        buf = shm.read()
        shm.detach()
        mutex.release()

        bgrImg = np.frombuffer(buf, np.uint8).reshape(
            OPTIONS.height, OPTIONS.width, OPTIONS.channels
        )
    except:
        if len(sys.argv) > 1:
            bgrImg = cv2.imread(sys.argv[1])
        else:
            print("No shared memory found and no file provided!")
            exit(1)

    stateMachine.runState(bgrImg)
    cv2.waitKey(0)
