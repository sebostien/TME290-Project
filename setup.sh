#!/bin/sh

cd ./opendlv-desktop-data/ || exit

mkdir tmp 2> /dev/null
cd tmp || exit
git clone https://git.opendlv.org/community/opendlv-tutorial-kiwitasks.git 2> /dev/null
git clone https://git.opendlv.org/community/opendlv-tutorial-kiwiperception.git 2> /dev/null

cp opendlv-tutorial-kiwitasks ../ -n -r
cp opendlv-tutorial-kiwiperception ../ -n -r

