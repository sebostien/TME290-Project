# TME290 Assignment

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
  - [ ] Drive between cones
  - [ ] Check for orange cones
  - [ ] Images of kiwi-car side profile. Needed for training of YOLOv3
  - [ ] Right-hand rule
- [ ] Task 4
  - [x] Park on blue paper
  - [x] Park on PostIt.
  - [ ] Global position to find its way back to blue paper and to know
        where it has already been.
- [ ] Task 5
  - [ ] Images of kiwi-car side profile. Needed for training of YOLOv3
  - [ ] Follow behind car (should be easy once above is done)

## Darknet YOLOv3-tiny

Clone and compile [darknet](https://github.com/AlexeyAB/darknet) into
`./darknet/` directory.

Run `darknet/train.sh` to continue training.

Using `width = 192` and `height = 128` to keep the kiwi-car's aspect ratio.
Large enough to get good accuracy with limited time of training (~3 hours).
