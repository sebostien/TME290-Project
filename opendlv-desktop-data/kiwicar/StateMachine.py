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
    MIN_PEDAL_POSITION,
    MAX_PEDAL_POSITION,
    WIGGLE_WHEELS_TURN,
    WIGGLE_WHEELS_MILLIS,
)
from util import Vec2, Region
from yolov3_tiny import Prediction, forwardDNN

import opendlv_standard_message_set_v0_9_10_pb2
import OD4Session
import cv2

CONE_MIN_AREA = OPTIONS.width * OPTIONS.height * 0.0002
CONE_MAX_AREA = OPTIONS.width * OPTIONS.height * 0.018
PUT_RECT_TEXT = True  # Shows information about regions found

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


def putText(
    img: np.ndarray, text: str, x: float, y: float, color: tuple[int, int, int]
):
    if PUT_RECT_TEXT:
        cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color)


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
            case State.NOTHING | State.BETWEEN_CONES | State.BETWEEN_CONES_WITH_CARS | State.DEBUG_COLORS | State.DRIVE_BEHIND_CAR:
                return
            case State.LOOK_FOR_PAPER:
                self.currentState = State.LOOK_FOR_POSTIT
                return
            case State.LOOK_FOR_POSTIT:
                self.currentState = State.LOOK_FOR_PAPER
                return
        print("Error: nextState unhandled case: ", self.currentState)
        exit(1)

    def runState(self, bgrImg: np.ndarray):
        # print(self.currentState)
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
                print(f"Min cone area: {CONE_MIN_AREA}")
                print(f"Max cone area: {CONE_MAX_AREA}")
                self.getBlueCones(hsvImg, outImg)
                self.getYellowCones(hsvImg, outImg)
                # self.getPaperPosition(hsvImg, outImg)
                # self.getPostItPosition(hsvImg, outImg)
            case State.BETWEEN_CONES_WITH_CARS:
                self.handleState_BETWEEN_CONES(hsvImg, bgrImg, outImg, enable_net=True)
            case State.BETWEEN_CONES:
                self.handleState_BETWEEN_CONES(hsvImg, bgrImg, outImg, enable_net=False)
            case State.LOOK_FOR_PAPER:
                self.handleState_LOOK_FOR_PAPER(hsvImg, outImg)
            case State.LOOK_FOR_POSTIT:
                self.handleState_LOOK_FOR_POSTIT(hsvImg, outImg)
            case State.DRIVE_BEHIND_CAR:
                prediction = self.getKiwiPredictions(bgrImg, outImg)
                if len(prediction) > 0:
                    # Found car
                    cx = (prediction[0].x1 + prediction[0].x2) // 2
                    cv2.circle(outImg, (cx, OPTIONS.height // 2), 4, (255, 0, 0), 4)
                    world = self.screenToWorld(cx, 0)
                    angle = self.targetToAngle(world)
                    self.sendSteerRequest(angle)
                    # Make sure we don't collide
                    if self.distFront < STOP_DISTANCE_FRONT:
                        print("Breaking behind car")
                        self.sendPedalRequest(0)
                    else:
                        print("Following car")
                        self.sendPedalRequest(MIN_PEDAL_POSITION)
                else:
                    # No car but something in-front
                    if self.distFront < STOP_DISTANCE_FRONT:
                        self.reverseOut()

                    # Look for a car
                    print("Looking for car")
                    self.sendPedalRequest(MIN_PEDAL_POSITION)
                    if self.distFront < 0.5:
                        # Turn
                        self.sendSteerRequest(-38)

        if DEBUG:
            imshow("Image", outImg)
            cv2.waitKey(1)

    # steerDegrees [-38, 38]
    def sendSteerRequest(self, steerDegrees: float):
        if MODE != Mode.RUNNING_ON_KIWI:
            return

        # print("Angle: ", steerDegrees)

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
        y = (screenX / self.width) * 2 - 1  # Correct

        # Convert y from [0, height] to x [0,1]
        x = -(screenY / self.height) + 1  # Correct
        return Vec2(x, y)

    # World coodrinate x: [0,1], y: [-1, 1]
    # Screen coodrinate: x: [0,width], y: [0, height]
    def worldToScreen(self, worldCoordinate: Vec2) -> tuple[int, int]:
        # World y [-1,1] becomes x [0, width]
        x = self.width / 2 + (self.width / 2) * worldCoordinate.y  # Correct
        # World x [0, 1] becomes y [0, height]
        y = self.height - self.height * worldCoordinate.x  # Correct
        return (int(x), int(y))

    def targetToAngle(self, target: Vec2) -> float:
        """
        Returns angle to target.
        target.y should be in [-1, 1]

        a----b
        |   /
        |  /     We want angle t
        |t/        tan(t) = ab / ac
        |/
        c
        """
        # Useless. Pretty much linear without target.x
        # https://www.wolframalpha.com/input?i=plot+arctan%28x%2F1.28%29+*+180+%2F+pi+from+-1+to+1
        # ab = target.y
        # ac = 1.28  # Angle is 38 when y=1
        # rad = np.arctan(ab / ac)
        # deg = rad * 180 / np.pi
        return target.y * 38

    def targetToPedal(self, steerAngle: float) -> float:
        """
        Limits the pedal position depending on the steering angle.
        """

        # Angle in range [-38, 38] deg
        # Steer max when angle is 0 and min when |angle| = 38
        # https://www.wolframalpha.com/input?i=plot+-%7Ctanh%28x+%2F+38+*+pi%29%5E2%7C+%2B+1+from+-38+to+38
        pedal = -np.tanh(steerAngle / 38 * np.pi) ** 2 + 1  # [0,1]
        return pedal * (MAX_PEDAL_POSITION - MIN_PEDAL_POSITION) + MIN_PEDAL_POSITION

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
        img = cv2.Canny(img, 30, 90, 3)

        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # imshow("Counturs", img)

        paper = Region(0, 0, 0)
        for contour in contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            approx = cv2.approxPolyDP(
                contour, 0.001 * cv2.arcLength(contour, True), True
            )
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
            putText(
                outImg,
                f"{len(approx)} ; {area}",
                x,
                y,
                (0, 0, 255),
            )
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
        img = cv2.erode(img, KERNEL_2_2, iterations=2)
        # imshow("PostIt erode 1", img)
        img = cv2.dilate(img, KERNEL_2_2, iterations=15)
        # imshow("PostIt dilate", img)
        img = cv2.erode(img, KERNEL_2_2, iterations=4)
        # imshow("PostIt erode 2", img)

        # Canny edge detection
        img = cv2.Canny(img, 30, 90, 3)

        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # imshow("Counturs" , canny )

        postIt = Region(0, 0, 0)
        for contour in contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            approx = cv2.approxPolyDP(
                contour, 0.001 * cv2.arcLength(contour, True), True
            )
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            putText(
                outImg,
                f"{len(approx)} ; {area}",
                x,
                y,
                (0, 0, 255),
            )
            if area > postIt.area:
                cv2.rectangle(outImg, (x, y), (x + w, y + h), POST_IT_RECTANGLE)
                postIt = Region(x + w / 2, y + h / 2, area)

        if postIt.area == 0:
            return None
        else:
            return postIt

    def getBlueCones(self, img: np.ndarray, outImg: np.ndarray) -> list[Region]:
        blueColors = cv2.inRange(img, OPTIONS.blueConeLow, OPTIONS.blueConeHigh)
        imshow("Blue cones", img)
        return self.getConePositions(blueColors, outImg, BLUE_CONES_RECTANGLE)

    def getYellowCones(self, img: np.ndarray, outImg: np.ndarray) -> list[Region]:
        yellowColors = cv2.inRange(img, OPTIONS.yellowConeLow, OPTIONS.yellowConeHigh)
        imshow("Yellow cones", img)
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
            approx = cv2.approxPolyDP(c, 0.001 * cv2.arcLength(c, True), True)
            area = cv2.contourArea(c)
            [x, y, w, h] = cv2.boundingRect(c)
            if (
                w > h * 1.2
                or h > w * 2
                or len(approx) <= 10
                or area < CONE_MIN_AREA
                or area > CONE_MAX_AREA
            ):
                continue

            putText(
                outImg,
                f"{len(approx)} ; {area}",
                x,
                y,
                (0, 0, 255),
            )
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

        # Check that the car is not blocked.
        if self.distFront < STOP_DISTANCE_FRONT:
            pedal = 0
        elif self.distFront < FULL_DISTANCE_FRONT:
            # Limit speed if front distance too short
            pedal = MIN_PEDAL_POSITION
        elif enable_net:
            # Check with YOLO
            prediction = self.getKiwiPredictions(bgrImg, outImg)
            if len(prediction) > 0:
                # Limit speed if car found
                pedal = MIN_PEDAL_POSITION

        self.sendPedalRequest(pedal)
        if DEBUG:
            # Map pedal to range [height/2, height]
            pedalY = (
                -(pedal - MIN_PEDAL_POSITION)
                / (MAX_PEDAL_POSITION - MIN_PEDAL_POSITION)
                * self.height
                / 2
                + self.height
            )
            cv2.circle(outImg, (goalScreen[0], int(pedalY)), 12, (255, 0, 0), -1)

    def handleState_LOOK_FOR_PAPER(self, hsvImg: np.ndarray, outImg: np.ndarray):
        if self.distFront < STOP_DISTANCE_FRONT:
            self.reverseOut()

        paper = self.getPaperPosition(hsvImg, outImg)
        if paper == None:
            # No postIt found
            # Continues in previous direction unless wall in-front
            if self.distFront < 0.5:
                self.sendSteerRequest(-38)
            self.sendPedalRequest(MIN_PEDAL_POSITION)
        else:
            cv2.rectangle(
                outImg,
                (0, int(0.8 * self.height)),
                (OPTIONS.width, int(0.8 * self.width)),
                (0, 0, 255),
                2,
            )
            if paper.mid.y > 0.8 * self.height:
                print("Parked on paper!")
                self.sendPedalRequest(MIN_PEDAL_POSITION)
                time.sleep(0.1) # TODO: Adjust time
                self.sendPedalRequest(0)
                self.wiggle_wheels()
                self.nextState()
            goal = self.screenToWorld(paper.mid.x, paper.mid.y)
            angle = self.targetToAngle(goal)
            self.sendSteerRequest(angle)
            self.sendPedalRequest(MIN_PEDAL_POSITION)

    def handleState_LOOK_FOR_POSTIT(self, hsvImg: np.ndarray, outImg: np.ndarray):
        if self.stateEntryTime > 1.5 * 60 * 1000:
            # Needs to find paper again.
            print("Time is up! Going back for blue paper!")
            self.nextState()
            return

        # Don't spam stdout
        if (
            self.stateEntryTime // 100 % 10 <= 1
            and self.stateEntryTime // 1000 % 5 == 0
        ):
            print(f"Time since last on paper: {self.stateEntryTime // 1000}s")

        if self.distFront < STOP_DISTANCE_FRONT:
            self.reverseOut()

        postIt = self.getPostItPosition(hsvImg, outImg)
        if not postIt:
            # No postIt found
            # Continues in previous direction unless wall in-front
            if self.distFront < 0.5:
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
                print("Parked on PostIt!")
                self.sendPedalRequest(MIN_PEDAL_POSITION)
                time.sleep(0.2) # TODO: Adjust time
                self.sendPedalRequest(0)
                # Car is on postIt
                self.wiggle_wheels()
                return
            goal = self.screenToWorld(postIt.mid.x, postIt.mid.y)
            angle = self.targetToAngle(goal)
            self.sendSteerRequest(angle)
            self.sendPedalRequest(MIN_PEDAL_POSITION)

    def reverseOut(self):
        print("Trying to reverse")
        print(f"Front distance {self.distFront}")
        print(f"Rear distance {self.distRear}")
        while self.distFront < 0.5:
            if (
                self.distFront < STOP_DISTANCE_FRONT
                and self.distRear < STOP_DISTANCE_FRONT
            ):
                print("STUCK! PLEASE MOVE ME")
                time.sleep(1)  # Limit stdout
                continue

            if self.distFront < STOP_DISTANCE_FRONT:
                # Blocked front. Try to reverse
                self.sendSteerRequest(38)
                self.sendPedalRequest(-0.5)
            else:
                # Blocked back. Drive forwards
                self.sendSteerRequest(-38)
                self.sendPedalRequest(MIN_PEDAL_POSITION)
            time.sleep(0.04)

        self.sendSteerRequest(0)
        self.sendPedalRequest(0)
        print("Done!\n")

    def wiggle_wheels(self):
        print("Wiggle Wheels!")
        self.sendPedalRequest(0)
        t = 0
        i = 0
        while i < 10:
            if i % 2 == 0:
                print("-20")
                self.sendSteerRequest(-20)
            else:
                print("20")
                self.sendSteerRequest(20)
            i += 1
            t += WIGGLE_WHEELS_TURN
            time.sleep(WIGGLE_WHEELS_TURN / 1000)
        print("Done!")
