import os
import shutil
import random

imgDir = "./build/darknet/x64/data/obj/"

trainTxt = "./build/darknet/x64/data/train.txt"
valTxt = "./build/darknet/x64/data/val.txt"
testTxt = "./build/darknet/x64/data/test.txt"

trainImages = []
valImages = []


for line in os.listdir(imgDir):
    if line.endswith(".jpg"):
        fileName = f"{imgDir}{line}"
        # 10% As validation
        if random.random() < 0.1:
            valImages.append(fileName)
        else:
            trainImages.append(fileName)

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
