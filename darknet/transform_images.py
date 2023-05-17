import os
from PIL import Image
import shutil
import random

dataDir = "../opendlv-desktop-data/opendlv-tutorial-kiwitasks/kiwi-detection/"
imgDir = "./build/darknet/x64/data/obj/"
# files = ["neg.txt", "pos_all.txt"]
files = ["pos_all.txt"]
trainTxt = "./build/darknet/x64/data/train.txt"
valTxt = "./build/darknet/x64/data/val.txt"
testTxt = "./build/darknet/x64/data/test.txt"

trainImages = []
valImages = []

val = 0
train = 0

try:
    os.mkdir(imgDir)
except:
    pass

for file in files:
    with open(dataDir + file, "r") as f:
        s = f.read()
        for line in s.splitlines():
            data = line.strip().split(" ")
            imgName = data[0].split("/")[1].split(".")[0]
            img = Image.open(dataDir + data[0], mode="r", formats=None)
            img.save(f"{imgDir}{imgName}.jpg")
            l = ""
            if len(data) == 6:
                # class x_center y_center width height
                x = int(data[2]) / 640
                y = int(data[3]) / 480
                w = int(data[4]) / 640
                h = int(data[5]) / 480
                l = f"0 {x + w / 2} {y + h / 2} {w} {h}"

            # 10% As validation
            if random.random() < 0.1:
                valImages.append(f"{imgDir}{imgName}.jpg")
            else:
                trainImages.append(f"{imgDir}{imgName}.jpg")

            with open(f"{imgDir}{imgName}.txt", "w") as f:
                f.write(l)

with open(trainTxt, "w") as f:
    for l in trainImages:
        f.writelines(f"{l}\n")

with open(valTxt, "w") as f:
    for l in valImages:
        f.writelines(f"{l}\n")

shutil.copy(valTxt, testTxt)

val = len(valImages)
train = len(trainImages)
total = val + train
print(f"Images: {total}")
print(f"Val:    {val:<5} {round(val/total * 100,1)}%")
print(f"Train:  {train:<5} {round(train/total * 100,1)}%")
