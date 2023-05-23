from enum import Enum
from typing import Any


class Mode(Enum):
    RUNNING_ON_KIWI = 1
    REC_FROM_TASKS = 2
    REC_FROM_KIWI = 3


class State(Enum):
    NOTHING = 0
    DEBUG_COLORS = 1
    BETWEEN_CONES = 2
    BETWEEN_CONES_WITH_CARS = 3
    LOOK_FOR_PAPER = 4
    LOOK_FOR_POSTIT = 5
    WIGGLE_WHEELS_THEN_POSTIT = 6
    WIGGLE_WHEELS_THEN_PAPER = 7
    DRIVE_BEHIND_CAR = 8
    GET_OUT_THEN_PAPER = 9
    GET_OUT_THEN_POSTIT = 10


################################################################################
# Global options

DEBUG = False
MODE = Mode.RUNNING_ON_KIWI
START_STATE = State.LOOK_FOR_PAPER

# Distance below this on front sensor will make the car stop
STOP_DISTANCE_FRONT = 0.2  # m
# Distance above this on front sensor will not limit the speed of the car
FULL_DISTANCE_FRONT = 0.5  # m

################################################################################
# Constants

import numpy as np


class Options:
    width: int
    height: int
    channels: int
    cid: int
    cameraName: str
    bluePaperLow: np.ndarray
    bluePaperHigh: np.ndarray
    greenPostItLow: np.ndarray
    greenPostItHigh: np.ndarray
    blueConeLow: np.ndarray
    blueConeHigh: np.ndarray
    yellowConeLow: np.ndarray
    yellowConeHigh: np.ndarray

    def __init__(self, d: dict[str, Any]):
        for k, v in d.items():
            setattr(self, k, v)


match MODE:
    case Mode.RUNNING_ON_KIWI:
        OPTIONS = Options(
            {
                "width": 640,
                "height": 480,
                "channels": 3,
                "cid": 140,
                "cameraName": "/tmp/img.bgr",
                "bluePaperLow": (90, 50, 50),
                "bluePaperHigh": (110, 255, 255),
                "greenPostItLow": (30, 50, 120),
                "greenPostItHigh": (45, 255, 255),
                "blueConeLow": (100, 50, 20),
                "blueConeHigh": (130, 255, 255),
                "yellowConeLow": (18, 50, 120),
                "yellowConeHigh": (30, 255, 255),
            }
        )
    case Mode.REC_FROM_TASKS:
        OPTIONS = Options(
            {
                "width": 1280,
                "height": 720,
                "channels": 4,
                "cid": 111,
                "cameraName": "/tmp/img.argb",
                "bluePaperLow": (90, 100, 50),
                "bluePaperHigh": (110, 200, 150),
                "greenPostItLow": (30, 50, 120),
                "greenPostItHigh": (45, 255, 255),
                "blueConeLow": (100, 50, 20),
                "blueConeHigh": (130, 255, 255),
                "yellowConeLow": (18, 50, 120),
                "yellowConeHigh": (30, 200, 255),
            }
        )
    case Mode.REC_FROM_KIWI:
        OPTIONS = Options(
            {
                "width": 640,
                "height": 480,
                "channels": 4,
                "cid": 111,
                "cameraName": "/tmp/img.argb",
                "bluePaperLow": (90, 100, 50),
                "bluePaperHigh": (110, 255, 255),
                "greenPostItLow": (30, 50, 120),
                "greenPostItHigh": (45, 255, 255),
                "blueConeLow": (100, 50, 20),
                "blueConeHigh": (130, 255, 255),
                "yellowConeLow": (18, 50, 120),
                "yellowConeHigh": (30, 200, 255),
            }
        )
