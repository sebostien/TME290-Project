# TME290 Assignment

## Submission details

The code for the kiwi-car is located in the [kiwicar](https://github.com/sebostien/TME290-Project/tree/main/opendlv-desktop-data/kiwicar) directory.

- [StateMachine.py](https://github.com/sebostien/TME290-Project/blob/main/opendlv-desktop-data/kiwicar/StateMachine.py)
  Contains all logic for the behaviour of each task.
- [options.py](https://github.com/sebostien/TME290-Project/blob/main/opendlv-desktop-data/kiwicar/options.py)
  Contains options to easily switch between environments.
- [yolov3_tiny.py](https://github.com/sebostien/TME290-Project/blob/main/opendlv-desktop-data/kiwicar/yolov3_tiny.py)
  Contains code for setup and usage of the YOLOv3 detection model.
- [runKiwiCar.py](https://github.com/sebostien/TME290-Project/blob/main/opendlv-desktop-data/kiwicar/runKiwiCar.py)
  Setup code for OD4Session and camera usage. Mostly provided by the tutorial repository.

Code for darknet training is located in the [darknet](https://github.com/sebostien/TME290-Project/tree/main/darknet) directory.

- [transform_images.py](https://github.com/sebostien/TME290-Project/blob/main/darknet/transform_images.py)
  Code for transforming course provided images into darknet format.
- [seperate_val_train.py](https://github.com/sebostien/TME290-Project/blob/main/darknet/seperate_val_train.py)
  Code for generating lists of training and validate groups.
- [kiwicarv3.cfg](https://github.com/sebostien/TME290-Project/blob/main/darknet/kiwicarv3.cfg)
  Config for our YOLOv3-tiny model.

## Darknet YOLOv3-tiny

Clone and compile [darknet](https://github.com/AlexeyAB/darknet) into
`./darknet/` directory.

Run `darknet/train.sh` to continue training (after adding images).

Using `width = 192` and `height = 128` to keep the kiwi-car's aspect ratio.
Large enough to get good accuracy with limited time of training (~3 hours).
