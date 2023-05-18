#!/bin/sh

./darknet detector train build/darknet/x64/data/obj.data kiwicarv3.cfg ./backup/kiwicarv3_final.weights -map -clear
