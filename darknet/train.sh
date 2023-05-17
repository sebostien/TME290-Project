#!/bin/sh

./darknet detector train build/darknet/x64/data/obj.data kiwicarv3.cfg ./backup/kiwicarv3_best.weights -map
