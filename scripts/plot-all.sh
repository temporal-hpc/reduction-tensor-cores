#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo ""
    echo "run as ${0} DEV-MODEL"
    echo "example #1: scripts/plot-all.sh TESLA-V100"
    echo "example #2: scripts/plot-all.sh INTEL-XEON-GOLD"
    echo ""
    exit;
fi
DEV=${1}

#for rconf and pconf, adjust the yranges manually
RECY1=0.26; RECY2=0.31
SINGLEY1=0.235; SINGLEY2=0.3
SPLITY1=0.2 SPLITY2=0.3

# (1) rconf and pconf
gnuplot -c scripts/plot-rconf.gp ${DEV} normal recurrence ${RECY1} ${RECY2}
gnuplot -c scripts/plot-rconf.gp ${DEV} normal singlepass ${SINGLEY1} ${SINGLEY2}
gnuplot -c scripts/plot-pconf.gp ${DEV} normal split ${SPLITY1} ${SPLITY2}



# (2) variants
gnuplot -c scripts/plot-variants-runtime.gp ${DEV} normal
gnuplot -c scripts/plot-variants-speedup.gp ${DEV} normal
gnuplot -c scripts/plot-variants-error.gp ${DEV} normal

gnuplot -c scripts/plot-variants-runtime.gp ${DEV} uniform
gnuplot -c scripts/plot-variants-speedup.gp ${DEV} uniform
gnuplot -c scripts/plot-variants-error.gp ${DEV} uniform




# (3) comparison with CUB and OpenMP
gnuplot -c scripts/plot-comparison-runtime.gp ${DEV} normal
gnuplot -c scripts/plot-comparison-speedup.gp ${DEV} normal
gnuplot -c scripts/plot-comparison-error.gp ${DEV} normal
gnuplot -c scripts/plot-comparison-beps.gp ${DEV} normal
#
gnuplot -c scripts/plot-comparison-runtime.gp ${DEV} uniform
gnuplot -c scripts/plot-comparison-speedup.gp ${DEV} uniform
gnuplot -c scripts/plot-comparison-error.gp ${DEV} uniform
gnuplot -c scripts/plot-comparison-beps.gp ${DEV} uniform
