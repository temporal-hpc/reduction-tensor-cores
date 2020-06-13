reset

gpu  = ARG1
dist = ARG2
cpu  = ARG3

print "plot-comparison-error.gp ---> GPU: ",gpu," dist: ",dist

out     = 'plots/comparison-error-'.gpu.'-'.dist.'.eps'
mytitle = "Error vs CUB Library (".gpu.")\n".dist." Distribution\n "

set autoscale # scale axes automatically
set term postscript eps color blacktext "Courier" 18
set output out
set title mytitle font "Courier, 22"

set ylabel 'Error %' rotate by 90 offset -0.3
set yrange [0.0000005:0.02]
set log y

set xlabel 'n x 10^{6}'
set xrange [0:108000000]
set font "Courier, 20"
set pointsize   1.0
set xtics format "%1.0s"
set ytics format "10^{%L}"
set key Left top right reverse samplen 3.0 font "Courier,18" spacing 1 

set style line 1 lt 1 lc rgb 'forest-green' dt 1    pt 5    pi -6   lw 2 # green   
set style line 2 lt 2 lc rgb 'black'       dt 2    pt 2    pi -6   lw 2 # orange
set style line 3 lt 3 lc rgb 'web-blue'     dt 6    pt 6    pi -6   lw 2 # blue
set style line 4 lt 4 lc rgb 'red'          dt 5    pt 11   pi -6   lw 2 # purple
set style line 5 lt 1 lc rgb '#77ac30'              pt 13   pi -6   lw 2 # green
set style line 6 lt 1 lc rgb '#4dbeee'              pt 4    pi -6   lw 2 # light-blue
set style line 7 lt 1 lc rgb '#a2142f'              pt 8    pi -6   lw 2 # red

singlepass = 'data/alg-singlepass-'.gpu.'-'.dist.'-B32.dat'
cub16 = 'data/alg-CUB-FP16-'.gpu.'-'.dist.'.dat'
cub32 = 'data/alg-CUB-FP32-'.gpu.'-'.dist.'.dat'
ompDouble = 'data/alg-omp-double-'.cpu.'-'.dist.'-B32.dat'

plot    singlepass  using 1:12 title "single-pass" with lp ls 1,\
        cub16       using 1:9 title "CUB (half)"  with lp ls 3,\
        cub32       using 1:9 title "CUB (float)" with p ls 2,\
        ompDouble   using 1:12 title "OpenMP (nt=10)" with p ls 4
