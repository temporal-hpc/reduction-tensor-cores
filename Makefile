BSIZE=256
R=1
DEBUG=DUMMY
POWER=DUMMY
POWER_DEBUG=DUMMY
KDEBUG=DUMMY
NPROC=16
PARAMS=-O3 -DNPROC=${NPROC} -DBSIZE=${BSIZE} -D${POWER} -DR=${R} -D${DEBUG} -D${KDEBUG} -D${POWER_DEBUG} --default-stream per-thread -lnvidia-ml -Xcompiler -lpthread,-fopenmp
ARCH=sm_75
SOURCES=$(wildcard src/*.cpp) $(wildcard src/*.cu)
all:
	nvcc ${PARAMS} -arch ${ARCH} ${SOURCES} -o bin/prog
