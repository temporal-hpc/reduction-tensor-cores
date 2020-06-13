#!/bin/bash
if [ "$#" -ne 18 ]; then
    echo "run as ${0} DEV OMPTHREADS  N1 N2 DN    B1 B2 DB  ARCH R FS DIST SEED    KREPEATS SAMPLES BINARY ALG OUTFILE"
    exit;
fi
DEV=$1
OMPTHREADS=$2
STARTN=$3
ENDN=$4
DN=$5
STARTB=$6
ENDB=$7
DB=$8
ARCH=$9
R=${10}
FS=${11}
DIST=${12}
SEED=${13}
REPEAT=${14}
SAMPLES=${15}
BINARY=${16}
ALG=${17}
OUTFILE=${18}
DISTRIBUTION=("normal" "uniform" "constant")
ALGORITHMS=("warpshuffle" "recurrence" "singlepass" "split" "omp-float" "omp-double")
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0

for B in `seq ${STARTB} ${DB} ${ENDB}`;
do
    CPU_AFF=$((${OMPTHREADS}-1))
    CPU_AFF="export GOMP_CPU_AFFINITY=0-${CPU_AFF}"
    COMPILE="make BSIZE=${B} ARCH=${ARCH} R=${R} NPROC=${OMPTHREADS}"
    echo "$COMPILE"
    C=`${COMPILE}`
    echo "${CPU_AFF}"
    `${CPU_AFF}`
    OUTPATH=data/alg-${ALGORITHMS[${ALG}]}-${OUTFILE}-${DISTRIBUTION[$DIST]}-B${B}.dat
    for N in `seq ${STARTN} ${DN} ${ENDN}`;
    do
        echo "[B=${B},R=${R},FS=${FS}]  ${N}"
        echo -n "${N}  ${B}  ${R}  ${FS}     " >> ${OUTPATH}
        M=0; S=0; x=0; y=0; z=0; v=0; w1=0; y1=0; z1=0; v1=0; w2=0; y2=0; z2=0; v2=0;
        for k in `seq 1 ${SAMPLES}`;
        do
            echo  "./${BINARY} ${DEV}    ${N} ${FS} ${REPEAT} ${SEED} ${DIST} ${ALG}"
            value=`./${BINARY} ${DEV}    ${N} ${FS} ${REPEAT} ${SEED} ${DIST} ${ALG}`
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
    echo " "
    echo " "
done
echo " "
