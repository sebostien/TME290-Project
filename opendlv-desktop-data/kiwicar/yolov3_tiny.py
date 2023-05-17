import cv2
import numpy as np

# TODO: Move
confThreshold = 0.05  # Confidence threshold
nmsThreshold = 0.2  # Non-maximum suppression threshold

CAR_RECTANGLE = (0, 0, 255)

netWidth = 192
netHeight = 128

kiwiNet = cv2.dnn.readNetFromDarknet("./kiwicarv3.cfg", "./kiwicarv3_best.weights")
kiwiNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
kiwiNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


class Prediction:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence

    def centerX(self) -> int:
        return (self.x1 + self.x2) // 2

    def centerY(self) -> int:
        return (self.y1 + self.y2) // 2


# img needs to be rgb
def forwardDNN(img: np.ndarray, outImage: np.ndarray) -> list[Prediction]:
    blob = cv2.dnn.blobFromImage(
        img, 1 / 255, (netWidth, netHeight), [0, 0, 0], 1, crop=False
    )
    kiwiNet.setInput(blob)
    output = kiwiNet.forward()

    t, _ = kiwiNet.getPerfProfile()
    print("Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency()))
    shape = img.shape
    prediction = handleOutput(shape[1], shape[0], output)
    for p in prediction:
        cv2.rectangle(outImage, (p.x1, p.y1), (p.x2, p.y2), CAR_RECTANGLE, 2)
    return prediction


def handleOutput(imgWidth: int, imgHeight: int, dnn_output) -> list[Prediction]:
    confidences = []
    boxes = []
    for detection in dnn_output:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if confidence > confThreshold:
            center_x = int(detection[0] * imgWidth)
            center_y = int(detection[1] * imgHeight)
            width = int(detection[2] * imgWidth)
            height = int(detection[3] * imgHeight)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    rects = []
    for i in indices:
        box = boxes[i]
        x1 = box[0]
        y1 = box[1]
        w = box[2]
        h = box[3]
        rects.append(Prediction(x1, y1, x1 + w, y1 + h, confidences[i]))

    return rects


if __name__ == "__main__":
    img = cv2.imread("./test.jpg")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = forwardDNN(rgb, img)
    cv2.imshow("Prediction:", img)
    cv2.waitKey(0)
