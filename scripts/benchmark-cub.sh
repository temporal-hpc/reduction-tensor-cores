#!/bin/bash
if [ "$#" -ne 11 ]; then
    echo "run as ${0} DEV   N1 N2 DN    DIST SEED    KREPEATS SAMPLES BINARY PREFIX OUTFILE"
    exit;
fi
DEV=$1
STARTN=$2
ENDN=$3
DN=$4
DIST=${5}
SEED=${6}
REPEAT=${7}
SAMPLES=${8}
BINARY=${9}
PREFIX=${10}
OUTFILE=${11}
DISTRIBUTION=("normal" "uniform")
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0

OUTPATH=data/alg-CUB-${PREFIX}-${OUTFILE}-${DISTRIBUTION[$DIST]}.dat
for N in `seq ${STARTN} ${DN} ${ENDN}`;
do
    echo "${BINARY}  ${N}"
    echo -n "${N}  ${B}  " >> ${OUTPATH}
    M=0; S=0; x=0; y=0; z=0; v=0; w1=0; y1=0; z1=0; v1=0; w2=0; y2=0; z2=0; v2=0;
    for k in `seq 1 ${SAMPLES}`;
    do
        echo  "${BINARY} ${DEV}    ${N} ${DIST} ${SEED} ${REPEAT}"
        value=`${BINARY} ${DEV}    ${N} ${DIST} ${SEED} ${REPEAT}`
        echo "${value}"
        x="$(cut -d',' -f1 <<<"$value")"
        y="$(cut -d',' -f2 <<<"$value")"
        w="$(cut -d',' -f3 <<<"$value")"
        z="$(cut -d',' -f4 <<<"$value")"
        v="$(cut -d',' -f5 <<<"$value")"
        w1=$(echo "scale=10; $w1+$w" | bc)
        y1=$(echo "scale=10; $y1+$y" | bc)
        z1=$(echo "scale=10; $z1+$z" | bc)
        v1=$(echo "scale=10; $v1+$v" | bc)
        oldM=$M;
        M=$(echo "scale=10;  $M+($x-$M)/$k"           | bc)
        S=$(echo "scale=10;  $S+($x-$M)*($x-${oldM})" | bc)
    done
    echo "done"
    MEAN=$M
    VAR=$(echo "scale=10; $S/(${SAMPLES}-1.0)"  | bc)
    STDEV=$(echo "scale=10; sqrt(${VAR})"       | bc)
    STERR=$(echo "scale=10; ${STDEV}/sqrt(${SAMPLES})" | bc)
    TMEAN[0]=${MEAN}
    TVAR[0]=${VAR}
    TSTDEV[0]=${STDEV}
    TSTERR[0]=${STERR}
    w2=$(echo "scale=10; $w1/$SAMPLES" | bc)
    y2=$(echo "scale=10; $y1/$SAMPLES" | bc)
    z2=$(echo "scale=10; $z1/$SAMPLES" | bc)
    v2=$(echo "scale=10; $v1/$SAMPLES" | bc)
    echo " "
    echo "---> (MEAN, VAR, STDEV, STERR, SUM, CPUSUM, #DIFF, %DIFF) -> (${TMEAN[0]}[ms], ${TVAR[0]}, ${TSTDEV[0]}, ${TSTERR[0]}, ${y2}, ${w2}, ${z2}, ${v2})"
    echo -n "${TMEAN[0]} ${TVAR[0]} ${TSTDEV[0]} ${TSTERR[0]} ${y} ${w} ${z} ${v}        " >> ${OUTPATH}
    echo " " >> ${OUTPATH}
done
echo " " >> ${OUTPATH}
echo " "
