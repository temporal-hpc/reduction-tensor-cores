#!/bin/sh
if [ "$#" -ne 11 ]; then
    echo "run as ${0} DEV ARCH STARTN ENDN DN DIST SEED KREPEATS SAMPLES BINARY OUTFILE"
    exit;
fi
DEV=$1
ARCH=$2
STARTN=$3
ENDN=$4
DN=$5
DIST=$6
SEED=$7
KREPEATS=${8}
SAMPLES=${9}
BINARY=${10}
OUTFILE=${11}
METHODS=("warpshuffle" "recurrence" "singlepass" "split")
NM=$((${#METHODS[@]}-1))

# these values are for TITAN-RTX
R=("1" "4" "4" "1")
FS=("0" "0" "0" "0.5")
BSIZE=("1024" "128" "32" "512")

# these values are for TESLA-V100
#R=("1" "5" "4" "1")
#FS=("0" "0" "0" "0.5")
#BSIZE=("1024" "32" "128" "512")

DISTRIBUTION=("normal" "uniform")

for i in $(seq 0 ${NM});
do
    echo "[EXECUTE] scripts/benchmark-alg.sh ${DEV} 1 ${OMPTHREADS}    ${STARTN} ${ENDN} ${DN}     ${BSIZE[$i]} ${BSIZE[$i]} 1     ${ARCH} ${R[$i]} ${FS[$i]} ${DIST} ${SEED}      ${KREPEATS} ${SAMPLES} ${BINARY} ${i} ${OUTFILE}"
    scripts/benchmark-alg.sh                 ${DEV} 1 ${OMPTHREADS}    ${STARTN} ${ENDN} ${DN}     ${BSIZE[$i]} ${BSIZE[$i]} 1     ${ARCH} ${R[$i]} ${FS[$i]} ${DIST} ${SEED}      ${KREPEATS} ${SAMPLES} ${BINARY} ${i} ${OUTFILE}
done
