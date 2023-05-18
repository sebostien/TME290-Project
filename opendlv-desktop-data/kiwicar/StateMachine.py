import numpy as np
import time
from options import (
    MODE,
    OPTIONS,
    DEBUG,
    State,
    Mode,
    STOP_DISTANCE_FRONT,
    FULL_DISTANCE_FRONT,
)
from util import Vec2, Region
from yolov3_tiny import Prediction, forwardDNN

import opendlv_standard_message_set_v0_9_10_pb2
import OD4Session
import cv2

WIGGLE_WHEELS_MILLIS = 2000  # Number of ms to wiggle wheels
MIN_PEDAL_POSITION = 0.11
MAX_PEDAL_POSITION = 0.16
CONE_MIN_AREA = OPTIONS.width * OPTIONS.height * 0.0002
CONE_MAX_AREA = OPTIONS.width * OPTIONS.height * 0.018

BLUE_CONES_RECTANGLE = (255, 0, 0)
YELLOW_CONES_RECTANGLE = (0, 255, 255)
PAPER_RECTANGLE = (255, 0, 0)
POST_IT_RECTANGLE = (0, 255, 0)

KERNEL_1_1 = np.ones((1, 1), np.uint8)
KERNEL_2_2 = np.ones((2, 2), np.uint8)
KERNEL_3_3 = np.ones((3, 3), np.uint8)
KERNEL_4_4 = np.ones((4, 4), np.uint8)


def imshow(name: str, img: np.ndarray):
    if DEBUG:
        cv2.imshow(name, img)


class StateMachine:
    currentState: State = State.NOTHING
    # ms since state entry
    stateEntryTime: int = 0
    session: OD4Session.OD4Session
    # Width of input img
    width: float
    # Height of input img
    height: float

    # Distance sensors
    distFront = 0
    distLeft = 0
    distRight = 0
    distRear = 0

    # For cone steering, in world coordinates
    prevTarget: Vec2

    def __init__(self, startState: State, session: OD4Session.OD4Session):
        self.currentState = startState
        self.stateEntryTime = 0
        self.session = session
        self.width = float(OPTIONS.width)
        self.height = float(OPTIONS.height)

        self.prevTarget = Vec2(0, 0)  # No pedal, Straight

    def incTime(self, dt: int):
        self.stateEntryTime += dt

    def onDistance(self, senderStamp: int, distance: float):
        #  print("Received distance; senderStamp= %s" % (str(senderStamp)))
        # print(
        #    "sent: %s, received: %s, sample time stamps: %s"
        #    % (str(timeStamps[0]), str(timeStamps[1]), str(timeStamps[2]))
        # )
        # print("%s" % (msg))
        if senderStamp == 0:
            self.distFront = distance
        if senderStamp == 1:
            self.distLeft = distance
        if senderStamp == 2:
            self.distRear = distance
        if senderStamp == 3:
            self.distRight = distance

    def nextState(self):
        """
        The state transition matrix.
        Will change the state immediately.
        """
        self.stateEntryTime = 0
        match self.currentState:
            case State.NOTHING | State.BETWEEN_CONES | State.BETWEEN_CONES_WITH_CARS | State.DEBUG_COLORS:
                return
            case State.LOOK_FOR_PAPER:
                self.currentState = State.WIGGLE_WHEELS_THEN_POSTIT
                return
            case State.LOOK_FOR_POSTIT:
                self.currentState = State.WIGGLE_WHEELS_THEN_PAPER
                return
            case State.WIGGLE_WHEELS_THEN_POSTIT:
                self.currentState = State.LOOK_FOR_POSTIT
                return
            case State.WIGGLE_WHEELS_THEN_PAPER:
                self.currentState = State.LOOK_FOR_PAPER
                return

    def runState(self, bgrImg: np.ndarray):
        # For new training images
        # millis = lambda: round(time.time() * 1000)
        # cv2.imwrite(f"./images/{millis()}.jpg", bgrImg)

        outImg = bgrImg.copy()
        hsvImg = self.getHSVImage(bgrImg)

        match self.currentState:
            case State.NOTHING:
                self.sendSteerRequest(0)
                self.sendPedalRequest(0)
            case State.DEBUG_COLORS:
                print(f"Min area: {CONE_MIN_AREA}, Max area: {CONE_MAX_AREA}")
                self.getBlueCones(hsvImg, outImg)
                self.getYellowCones(hsvImg, outImg)
                self.getPaperPosition(hsvImg, outImg)
                self.getPostItPosition(hsvImg, outImg)
            case State.BETWEEN_CONES_WITH_CARS:
                self.handleState_BETWEEN_CONES(hsvImg, bgrImg, outImg, enable_net=True)
            case State.BETWEEN_CONES:
                self.handleState_BETWEEN_CONES(hsvImg, bgrImg, outImg, enable_net=False)
            case State.LOOK_FOR_PAPER:
                self.handleState_LOOK_FOR_PAPER(hsvImg, outImg)
            case State.LOOK_FOR_POSTIT:
                self.handleState_LOOK_FOR_POSTIT(hsvImg, outImg)
            case State.WIGGLE_WHEELS_THEN_PAPER | State.WIGGLE_WHEELS_THEN_POSTIT:
                if self.stateEntryTime > WIGGLE_WHEELS_MILLIS:
                    self.nextState()
                self.sendPedalRequest(0)
                # Change direction of wheels every 500 ms
                self.sendSteerRequest(
                    10 if self.stateEntryTime // 500 % 2 == 0 else -10
                )

        if DEBUG:
            imshow("Image", outImg)
            cv2.waitKey(1)

    # steerDegrees [-38, 38]
    def sendSteerRequest(self, steerDegrees: float):
        if MODE != Mode.RUNNING_ON_KIWI:
            return

        groundSteeringRequest = (
            opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_GroundSteeringRequest()
        )
        steerRad = steerDegrees / 180 * np.pi
        groundSteeringRequest.groundSteering = steerRad
        self.session.send(1090, groundSteeringRequest.SerializeToString())

    # pedalPosition [-1.0, 0.25]
    def sendPedalRequest(self, pedalPosition: float):
        if MODE != Mode.RUNNING_ON_KIWI:
            return

        pedalPositionRequest = (
            opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_PedalPositionRequest()
        )
        pedalPositionRequest.position = pedalPosition
        self.session.send(1086, pedalPositionRequest.SerializeToString())

    # Screen coordinate [x,y]
    # Return world coordinate [x,y], x: [0, 1], y: [-1,1]
    def screenToWorld(self, screenX: float, screenY: float) -> Vec2:
        # Convert x from [0, width] to y [-1,1]
        y = (screenX / self.width) * 2 - 1

        # Convert y from [0, height] to x [0,1]
        x = -(screenY / self.height) + 1
        return Vec2(x, y)

    # World coodrinate x: [0,1], y: [-1, 1]
    def worldToScreen(self, worldCoordinate: Vec2) -> tuple[int, int]:
        x = self.width / 2 + (self.width / 2) * worldCoordinate.y
        y = self.height - self.height * worldCoordinate.x

        return (int(x), int(y))

    # Returns angle to target
    def targetToAngle(self, target: Vec2) -> float:
        return target.y * 38

    def targetToPedal(self, steerAngle: float) -> float:
        # TODO: Add distance to car in-front

        # Angle in range [-38, 38] deg
        # Steer max when angle is 0 and min when |angle| = 38
        # https://www.wolframalpha.com/input?i=plot+-%7Ctanh%28x+%2F+38+*+pi%29%5E2%7C+%2B+1+from+-38+to+38
        return (-np.tanh(steerAngle / 38 * np.pi) ** 2 + 1) * (
            MAX_PEDAL_POSITION - MIN_PEDAL_POSITION
        ) + MIN_PEDAL_POSITION

    def getHSVImage(self, bgrImg: np.ndarray) -> np.ndarray:
        """
        Returns hsv image with some regions blacked out
        """
        hsv = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2HSV)

        # Remove car
        cv2.rectangle(
            hsv,
            (int(OPTIONS.width * (1 / 7)), int(OPTIONS.height * 0.8)),
            (int(OPTIONS.width * (6 / 7)), OPTIONS.height),
            (0, 0, 0),
            -1,
        )
        # Remove top half
        cv2.rectangle(hsv, (0, 0), (OPTIONS.width, OPTIONS.height // 2), (0, 0, 0), -1)

        return hsv

    def getKiwiPredictions(
        self, bgrImg: np.ndarray, outImg: np.ndarray
    ) -> list[Prediction]:
        rgb = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        return forwardDNN(rgb, outImg)

    def getPaperPosition(self, hsvImg: np.ndarray, outImg: np.ndarray) -> Region | None:
        img = cv2.inRange(hsvImg, OPTIONS.bluePaperLow, OPTIONS.bluePaperHigh)
        # imshow("Paper colors", img)
        img = cv2.dilate(img, KERNEL_2_2, iterations=10)
        img = cv2.erode(img, KERNEL_1_1, iterations=15)

        # Canny edge detection
        edges = 30
        threashold1 = 90
        threashold2 = 3
        canny = cv2.Canny(img, edges, threashold1, threashold2)

        contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # imshow("Counturs" , canny )

        paper = Region(0, 0, 0)
        for contour in contours:
            # peri = cv2.arcLength(contour, True)
            # area = cv2.contourArea(contour)
            [x, y, w, h] = cv2.boundingRect(contour)
            area = w * h
            print(area)
            if area > 1000:  # Also a guess, should be tweaked
                if area > paper.area:
                    cv2.rectangle(outImg, (x, y), (x + w, y + h), PAPER_RECTANGLE)
                    paper = Region(x + w / 2, y + h / 2, area)

        if paper.area == 0:
            return None
        else:
            return paper

    def getPostItPosition(
        self, hsvImg: np.ndarray, outImg: np.ndarray
    ) -> Region | None:
        img = cv2.inRange(hsvImg, OPTIONS.greenPostItLow, OPTIONS.greenPostItHigh)
        # imshow("PostIt color", img)
        img = cv2.dilate(img, KERNEL_2_2, iterations=10)
        # imshow("PostIt dilate", img)
        img = cv2.erode(img, KERNEL_1_1, iterations=15)
        # imshow("PostIt erode", img)

        # Canny edge detection
        edges = 30
        threashold1 = 90
        threashold2 = 3
        canny = cv2.Canny(img, edges, threashold1, threashold2)

        contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # imshow("Counturs" , canny )

        postIt = Region(0, 0, 0)
        for contour in contours:
            # peri = cv2.arcLength(contour, True)
            # area = cv2.contourArea(contour)
            [x, y, w, h] = cv2.boundingRect(contour)
            area = w * h
            print(area)
            if area > 1000:  # Also a guess, should be tweaked
                if area > postIt.area:
                    cv2.rectangle(outImg, (x, y), (x + w, y + h), POST_IT_RECTANGLE)
                    postIt = Region(x + w / 2, y + h / 2, area)

        if postIt.area == 0:
            return None
        else:
            return postIt

    def getBlueCones(self, img: np.ndarray, outImg: np.ndarray) -> list[Region]:
        blueColors = cv2.inRange(img, OPTIONS.blueConeLow, OPTIONS.blueConeHigh)
        # imshow("Blue cones", img)
        return self.getConePositions(blueColors, outImg, BLUE_CONES_RECTANGLE)

    def getYellowCones(self, img: np.ndarray, outImg: np.ndarray) -> list[Region]:
        yellowColors = cv2.inRange(img, OPTIONS.yellowConeLow, OPTIONS.yellowConeHigh)
        # imshow("Yellow cones", img)
        return self.getConePositions(yellowColors, outImg, YELLOW_CONES_RECTANGLE)

    def getConePositions(
        self, img, outImg, rectColor: tuple[int, int, int]
    ) -> list[Region]:
        # imshow("Cone color", img)
        img = cv2.erode(img, KERNEL_2_2, iterations=5)
        # imshow("Erode", img)
        img = cv2.dilate(img, KERNEL_3_3, iterations=10)
        # imshow("Dilate", img)
        img = cv2.Canny(img, 30, 90, 3)
        # imshow("Canny", img)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cones: list[Region] = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.0001 * cv2.arcLength(c, True), True)
            area = cv2.contourArea(c)
            [x, y, w, h] = cv2.boundingRect(c)
            # cv2.putText(
            #     outImg,
            #     f"{len(approx)} ; {area}",
            #     (int(x), int(y)),
            #     cv2.FONT_HERSHEY_COMPLEX,
            #     0.5,
            #     (0, 0, 255),
            # )
            if (
                w > h * 1.2
                or h > w * 2
                or len(approx) <= 10
                or area < CONE_MIN_AREA
                or area > CONE_MAX_AREA
            ):
                continue
            [x, y, w, h] = cv2.boundingRect(c)
            cones.append(Region(x + w / 2, y + h / 2, area))
            cv2.rectangle(outImg, (x, y), (x + w, y + h), rectColor)
        return cones

    def betweenConeTarget(self, hsvImg: np.ndarray, outImg: np.ndarray) -> Vec2:
        blueCones = self.getBlueCones(hsvImg, outImg)
        yellowCones = self.getYellowCones(hsvImg, outImg)

        leftCones: list[float] = list(map(lambda cone: cone.mid.x, yellowCones))
        rightCones: list[float] = list(map(lambda cone: cone.mid.x, blueCones))

        leftMean = np.mean(leftCones or [0])
        rightMean = np.mean(rightCones or [self.width])

        # Make sure they don't cross over if detection is bad
        leftMean = min(leftMean, rightMean - self.width / 8)
        rightMean = max(rightMean, leftMean + self.width / 8)

        cv2.line(
            outImg,
            (int(leftMean), 0),
            (int(leftMean), OPTIONS.height),
            YELLOW_CONES_RECTANGLE,
            2,
        )
        cv2.line(
            outImg,
            (int(rightMean), 0),
            (int(rightMean), OPTIONS.height),
            BLUE_CONES_RECTANGLE,
            2,
        )

        middle = (leftMean + rightMean) / 2
        target = self.screenToWorld(middle, self.height / 2)
        # TODO: 0.5 makes the car not steer too much between frames. Is this too low/high?
        target.y = self.prevTarget.y + (target.y - self.prevTarget.y) * 0.5
        self.prevTarget = target
        return target

    ###########################################################
    # Handle specific states below

    def handleState_BETWEEN_CONES(
        self,
        hsvImg: np.ndarray,
        bgrImg: np.ndarray,
        outImg: np.ndarray,
        enable_net: bool,
    ):
        target = self.betweenConeTarget(hsvImg, outImg)
        goalScreen = self.worldToScreen(target)

        angle = self.targetToAngle(target)
        self.sendSteerRequest(angle)
        pedal = self.targetToPedal(angle)

        print(f"Distance front: {self.distFront}")

        # Check that the car is not blocked.
        # Only checks with the net if necessary
        if self.distFront > 0 and self.distFront < STOP_DISTANCE_FRONT:
            pedal = 0
        else:
            # Limit speed if front distance too short
            if self.distFront > 0 and self.distFront < FULL_DISTANCE_FRONT:
                pedal = MIN_PEDAL_POSITION
            if enable_net:
                # Check net
                prediction = self.getKiwiPredictions(bgrImg, outImg)
                if len(prediction) > 0:
                    print(prediction[0].x1)
                    # Limit speed if car found
                    pedal = MIN_PEDAL_POSITION
        self.sendPedalRequest(pedal)
        pedalY = (
            -(pedal - MIN_PEDAL_POSITION)
            / (MAX_PEDAL_POSITION - MIN_PEDAL_POSITION)
            * self.height
            / 2
            + self.height
        )
        cv2.circle(outImg, (goalScreen[0], int(pedalY)), 12, (255, 0, 0), -1)

    def handleState_LOOK_FOR_PAPER(self, hsvImg: np.ndarray, outImg: np.ndarray):
        paper = self.getPaperPosition(hsvImg, outImg)
        if paper == None:
            # No paper found
            # TODO: Drive in circle?
            # TODO: Break if near wall
            self.sendSteerRequest(-38)
            self.sendPedalRequest(MIN_PEDAL_POSITION)
        else:
            # Found paper
            cv2.rectangle(
                outImg,
                (0, int(0.8 * self.height)),
                (OPTIONS.width, int(0.8 * self.width)),
                (0, 0, 255),
                2,
            )
            if paper.mid.y > 0.8 * self.height:
                # Car is on paper
                self.nextState()
                self.sendSteerRequest(0)
                self.sendPedalRequest(0)
            goal = self.screenToWorld(paper.mid.x, paper.mid.y)
            angle = self.targetToAngle(goal)
            self.sendSteerRequest(angle)
            self.sendPedalRequest(MIN_PEDAL_POSITION)

    def handleState_LOOK_FOR_POSTIT(self, hsvImg: np.ndarray, outImg: np.ndarray):
        # Need to look for paper. Right now looks after 2 min
        if self.stateEntryTime > 2 * 60 * 1000:
            # Need to find paper
            self.currentState = State.LOOK_FOR_PAPER
            return

        postIt = self.getPostItPosition(hsvImg, outImg)
        if postIt == None:
            # No postIt found
            # TODO: Drive in circle?
            # TODO: Break if near wall
            self.sendSteerRequest(-38)
            self.sendPedalRequest(MIN_PEDAL_POSITION)
        else:
            # Found postIt
            # Middle below this line == on postIt
            cv2.rectangle(
                outImg,
                (0, int(0.8 * self.height)),
                (OPTIONS.width, int(0.8 * self.width)),
                (0, 0, 255),
                2,
            )
            if postIt.mid.y > 0.8 * self.height:
                # Car is on postIt
                self.nextState()
                self.sendSteerRequest(0)
                self.sendPedalRequest(0)
                return
            goal = self.screenToWorld(postIt.mid.x, postIt.mid.y)
            angle = self.targetToAngle(goal)
            self.sendSteerRequest(angle)
            self.sendPedalRequest(MIN_PEDAL_POSITION)
