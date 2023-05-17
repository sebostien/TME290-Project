#!/bin/sh

./darknet detector train build/darknet/x64/data/obj.data kiwicarv3.cfg ./yolov3-tiny.weights -map -clear
