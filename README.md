# TME290 Assignment

## Submission details

The code for the kiwi-car is located in `./opendlv-desktop-data/kiwicar/`.

The file `./opendlv-desktop-data/kiwicar/StateMachine.py` contains all logic for the behaviour of each task.
The file `./opendlv-desktop-data/kiwicar/yolov3_tiny.py` contains code to run YOLOv3 detection.
The file `./opendlv-desktop-data/kiwicar/` contains the setup code for camera and OD4Session.

## TODO

- [ ] Other
  - [x] Drive between cones.
  - [ ] Recording in demo room to check colors.
- [x] Task 1
  - [x] Drive between cones.
- [ ] Task 2
  - [x] Drive between cones.
  - [x] Stop with sensor.
  - [x] Stop with YOLOv3
  - [ ] Test with other cars
- [ ] Task 3
  - [x] Drive between cones
  - [ ] Check for orange cones
  - [ ] Images of kiwi-car side profile. Needed for training of YOLOv3
  - [ ] Right-hand rule
- [ ] Task 4
  - [x] Park on blue paper
  - [x] Park on PostIt.
  - [x] Global position to find its way back to blue paper and to know
        where it has already been.
        (Skipped this, driving left-turns instead)
- [ ] Task 5
  - [ ] Images of kiwi-car side profile. Needed for training of YOLOv3
        (Skipped, driving left-turns until rear of car found)
  - [x] Follow behind car (should be easy once above is done)

## Darknet YOLOv3-tiny

Clone and compile [darknet](https://github.com/AlexeyAB/darknet) into
`./darknet/` directory.

Run `darknet/train.sh` to continue training (after adding images).

Using `width = 192` and `height = 128` to keep the kiwi-car's aspect ratio.
Large enough to get good accuracy with limited time of training (~3 hours).
