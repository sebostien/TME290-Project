import numpy as np
from options import MODE, OPTIONS, DEBUG, State, Mode
from util import Vec2, Region
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

KERNEL_1_1 = np.ones((1, 1), np.uint8)
KERNEL_2_2 = np.ones((2, 2), np.uint8)
KERNEL_3_3 = np.ones((3, 3), np.uint8)
KERNEL_4_4 = np.ones((4, 4), np.uint8)


def imshow(name: str, img: np.ndarray):
    if DEBUG:
        cv2.imshow(name, img)


class StateMachine:
    currentState: State = State.NOTHING
    stateEntryTime: int = 0
    session: OD4Session.OD4Session
    width: float
    height: float

    # Distance sensors
    distances = {"front": 0.0, "left": 0.0, "right": 0.0, "rear": 0.0}

    # For cone steering, in world coordinates
    prevTarget: Vec2

    def __init__(self, startState: State, session: OD4Session.OD4Session):
        self.currentState = startState
        # ms since state entry
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
            self.distances["front"] = distance
        if senderStamp == 1:
            self.distances["left"] = distance
        if senderStamp == 2:
            self.distances["rear"] = distance
        if senderStamp == 3:
            self.distances["right"] = distance

    def nextState(self):
        self.stateEntryTime = 0
        match self.currentState:
            case State.NOTHING:
                return
            case State.BETWEEN_CONES:
                return
            case State.LOOK_FOR_PAPER:
                self.currentState = State.WIGGLE_WHEELS_THEN_GREEN
                return
            case State.WIGGLE_WHEELS_THEN_GREEN:
                self.currentState = State.LOOK_FOR_POSTIT
                return
            case State.WIGGLE_WHEELS_THEN_BLUE:
                self.currentState = State.LOOK_FOR_PAPER

    def runState(self, hsvImage, outImage):
        match self.currentState:
            case State.NOTHING:
                self.sendSteerRequest(0)
                self.sendPedalRequest(0)
            case State.DEBUG_COLORS:
                print(f"Min area: {CONE_MIN_AREA}, Max area: {CONE_MAX_AREA}")
                blueImage = cv2.inRange(
                    hsvImage, OPTIONS.blueConeLow, OPTIONS.blueConeHigh
                )
                self.getConePositions(blueImage, outImage, BLUE_CONES_RECTANGLE)
                yellowImage = cv2.inRange(
                    hsvImage, OPTIONS.yellowConeLow, OPTIONS.yellowConeHigh
                )
                # self.getConePositions(yellowImage, outImage, YELLOW_CONES_RECTANGLE)
                paperImage = cv2.inRange(
                    hsvImage, OPTIONS.bluePaperLow, OPTIONS.bluePaperHigh
                )
                # self.getPaperPosition(paperImage, outImage)
            case State.BETWEEN_CONES:
                blueImage = cv2.inRange(
                    hsvImage, OPTIONS.blueConeLow, OPTIONS.blueConeHigh
                )
                yellowImage = cv2.inRange(
                    hsvImage, OPTIONS.yellowConeLow, OPTIONS.yellowConeHigh
                )
                blueCones = self.getConePositions(
                    blueImage, outImage, BLUE_CONES_RECTANGLE
                )
                yellowCones = self.getConePositions(
                    yellowImage, outImage, YELLOW_CONES_RECTANGLE
                )

                leftCone: float = 0  # Yellow
                rightCone: float = self.width  # Blue

                for cone in blueCones:
                    if cone.mid.x < rightCone:
                        rightCone = cone.mid.x

                for cone in yellowCones:
                    if cone.mid.x > leftCone:
                        leftCone = cone.mid.x

                # Make sure they don't cross over
                rightCone = max(rightCone, leftCone + self.width / 8)
                leftCone = min(rightCone, leftCone - self.width / 8)

                cv2.line(
                    outImage,
                    (int(leftCone), 0),
                    (int(leftCone), OPTIONS.height),
                    YELLOW_CONES_RECTANGLE,
                    2,
                )
                cv2.line(
                    outImage,
                    (int(rightCone), 0),
                    (int(rightCone), OPTIONS.height),
                    BLUE_CONES_RECTANGLE,
                    2,
                )

                middle = (leftCone + rightCone) / 2
                target = self.screenToWorld(middle, self.height / 2)
                target.y = self.prevTarget.y + (target.y - self.prevTarget.y) * 0.5
                self.prevTarget = target
                goalScreen = self.worldToScreen(target)

                angle = self.targetToAngle(target)
                self.sendSteerRequest(angle)
                pedal = self.targetToPedal(angle)
                self.sendPedalRequest(pedal)
                pedalY = (
                    -(pedal - MIN_PEDAL_POSITION)
                    / (MAX_PEDAL_POSITION - MIN_PEDAL_POSITION)
                    * self.height
                    / 2
                    + self.height
                )
                cv2.circle(outImage, (goalScreen[0], int(pedalY)), 12, (255, 0, 0), -1)

            case State.LOOK_FOR_PAPER:
                paperImage = cv2.inRange(
                    hsvImage, OPTIONS.bluePaperLow, OPTIONS.bluePaperHigh
                )
                paper = self.getPaperPosition(paperImage, outImage)
                # imshow("Paper blue", paperImage)
                if paper == None:
                    # No paper found = Stop
                    # TODO: Drive in circle?
                    self.sendSteerRequest(0)
                    self.sendPedalRequest(0)
                else:
                    # Found paper
                    goal = self.screenToWorld(paper.mid.x, paper.mid.y)
                    angle = self.targetToAngle(goal)
                    self.sendSteerRequest(angle)
                    self.sendPedalRequest(MIN_PEDAL_POSITION)
            case State.WIGGLE_WHEELS_THEN_BLUE:
                if self.stateEntryTime > WIGGLE_WHEELS_MILLIS:
                    self.nextState()
                self.sendPedalRequest(0)
                self.sendSteerRequest(
                    10 if self.stateEntryTime // 500 % 2 == 0 else -10
                )
            case State.WIGGLE_WHEELS_THEN_GREEN:
                self.sendPedalRequest(0)
                self.sendSteerRequest(
                    10 if self.stateEntryTime // 500 % 2 == 0 else -10
                )

        if DEBUG:
            imshow("Image", outImage)
            cv2.waitKey(2)

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

    def getPaperPosition(self, img, outImage) -> Region | None:
        dilate = cv2.dilate(img, KERNEL_2_2, (-1, -1), iterations=10)
        erode = cv2.erode(dilate, KERNEL_1_1, (-1, 0), iterations=15)

        # Canny edge detection
        edges = 30
        threashold1 = 90
        threashold2 = 3
        canny = cv2.Canny(erode, edges, threashold1, threashold2)

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
                    cv2.rectangle(outImage, (x, y), (x + w, y + h), PAPER_RECTANGLE)
                    paper = Region(x + w / 2, y + h / 2, area)

        if paper.area == 0:
            return None
        else:
            return paper

    def getConePositions(
        self, img, outImage, rectColor: tuple[int, int, int]
    ) -> list[Region]:
        imshow("Cone color", img)
        img = cv2.erode(img, KERNEL_2_2, (-1, -1), iterations=5)
        # imshow("Erode", img)
        img = cv2.dilate(img, KERNEL_3_3, (-1, -1), iterations=10)
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
            #     outImage,
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
            cv2.rectangle(outImage, (x, y), (x + w, y + h), rectColor)
        return cones
