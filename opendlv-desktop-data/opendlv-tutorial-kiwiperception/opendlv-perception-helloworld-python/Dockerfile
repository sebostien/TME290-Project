# Copyright (C) 2023 Ola Benderius
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y \
    build-essential \
    cmake \
    software-properties-common \
    libopencv-dev

RUN add-apt-repository 'ppa:chrberger/libcluon' && \
    apt-get update && \
    apt-get install libcluon

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3-protobuf \
        python3-sysv-ipc \        
        python3-numpy \
        python3-opencv \
        protobuf-compiler
RUN apt-get clean

ADD . /opt/sources
WORKDIR /opt/sources
RUN make

ENTRYPOINT ["/opt/sources/opendlv-perception-helloworld.py"]
