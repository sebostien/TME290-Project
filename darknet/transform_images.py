from PIL import Image

dataPath = "../opendlv-desktop-data/opendlv-tutorial-kiwitasks/kiwi-detection/"
outFile = "./build/darknet/x64/data/obj/"
files = ["neg.txt", "pos_all.txt"]
trainTxt = "./build/darknet/x64/data/train.txt"

allImages = []

for file in files:
    with open(dataPath + file, "r") as f:
        s = f.read()
        for line in s.splitlines():
            line = line.split(" ")
            img_name = line[0].split("/")[1].split(".")[0]
            img = Image.open(dataPath + line[0], mode="r", formats=None)
            img.save(
                outFile + img_name + ".jpg",
            )
            allImages.append(f"build/darknet/x64/data/obj/{img_name}.jpg")
            l = ""
            if len(line) == 6:
                l = f"0 {int(line[2]) / 640} {int(line[3]) / 480} {int(line[4]) / 640} {int(line[5]) / 480}"
            with open(outFile + f"{img_name}.txt", "w") as f:
                f.write(l)

with open(trainTxt, "w") as f:
    for l in allImages:
        f.writelines(f"{l}\n")
