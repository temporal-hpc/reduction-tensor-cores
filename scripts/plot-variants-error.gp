reset

#if (!exists(ARG1) || !exists(ARG2)){
#    print "run as : gnuplot -c script.gp   GPU-MODEL    DISTRIBUTION"
#    exit
#}

gpu  = ARG1
dist = ARG2

print "plot-variants-error.gp ---> GPU: ",gpu," dist: ",dist

out = 'plots/variants-error-'.gpu.'-'.dist.'.eps'
title = "Variants Error vs CPU-FP64\n".dist." Distribution"

set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Courier" 24
set output out
set title title

set ytics mirror
set ylabel 'Error %' rotate by 90 offset -0.3
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set yrange [0.000001:0.05]
set xrange [0:107000000]
set log y

set xlabel 'n x 10^{6}'
set font "Courier, 20"
set pointsize   1.0
set xtics format "%1.0s"
set ytics format "10^{%L}"
set key right top font "Courier, 18"

set style line 1 lt 1 lc rgb 'forest-green' dt 1    pt 5    pi -6   lw 2 # green   
set style line 2 lt 2 lc rgb 'black'       dt 2    pt 2    pi -6   lw 2 # orange
set style line 3 lt 3 lc rgb 'web-blue'     dt 6    pt 6    pi -6   lw 2 # blue
set style line 4 lt 4 lc rgb 'red'          dt 5    pt 11   pi -6   lw 2 # purple
set style line 5 lt 1 lc rgb '#77ac30'              pt 13   pi -6   lw 2 # green
set style line 6 lt 1 lc rgb '#4dbeee'              pt 4    pi -6   lw 2 # light-blue
set style line 7 lt 1 lc rgb '#a2142f'              pt 8    pi -6   lw 2 # red

# variables
warp_shuffle_data = 'data/alg-warpshuffle-'.gpu.'-'.dist.'-B1024.dat'
split_data = 'data/alg-split-'.gpu.'-'.dist.'-B512.dat'
recurrence_data = 'data/alg-recurrence-'.gpu.'-'.dist.'-B32.dat'
single_pass_data = 'data/alg-singlepass-'.gpu.'-'.dist.'-B128.dat'

#print "warp_shuffle_data: ".warp_shuffle_data
#print "split_data: ".split_data
#print "recurrence_data: ".recurrence_data
#print "single_pass_data: ".single_pass_data

plot    warp_shuffle_data 	using 1:12 title "warp-shuffle" 	with lp ls 2,\
        recurrence_data 	using 1:12 title "v1-recurrence" 	with lp ls 3,\
        single_pass_data 	using 1:12 title "v2-single-pass" 	with lp ls 1,\
        split_data 		using 1:12 title "v3-split" 		with lp ls 4
