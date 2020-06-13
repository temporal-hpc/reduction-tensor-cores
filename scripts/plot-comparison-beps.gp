reset

gpu  = ARG1
dist = ARG2
cpu  = ARG3

print "plot-comparison-beps.gp ---> GPU: ",gpu," dist: ",dist

out     = 'plots/comparison-BEPS-'.gpu.'-'.dist.'.eps'
mytitle = "Billion Elements per Second (BEPS), ".gpu."\n".dist." Distribution\n "

set autoscale # scale axes automatically
set term postscript eps color blacktext "Courier" 18
set output out
set title mytitle font "Courier, 20"

set ylabel 'BEPS' rotate by 90 offset 0

set xlabel 'n x 10^{6}'
set xrange [0:108000000]
set font "Courier, 20"
set pointsize   1.0
set xtics format "%1.0s"
set key Left bot right reverse samplen 3.0 font "Courier,18" spacing 1.0 
set key at 105000000, 30

set style line 1 lt 1 lc rgb 'forest-green' dt 1    pt 5    pi -6   lw 2 # green   
set style line 2 lt 2 lc rgb 'black'       dt 2    pt 2    pi -6   lw 2 # orange
set style line 3 lt 3 lc rgb 'web-blue'     dt 6    pt 6    pi -6   lw 2 # blue
set style line 4 lt 4 lc rgb 'red'          dt 5    pt 11   pi -6   lw 2 # purple
set style line 5 lt 1 lc rgb '#77ac30'              pt 13   pi -6   lw 2 # green
set style line 6 lt 1 lc rgb '#4dbeee'              pt 4    pi -6   lw 2 # light-blue
set style line 7 lt 1 lc rgb '#a2142f'              pt 8    pi -6   lw 2 # red

singlepass  = 'data/alg-singlepass-'.gpu.'-'.dist.'-B32.dat'
cub16       = 'data/alg-CUB-FP16-'.gpu.'-'.dist.'.dat'
cub32       = 'data/alg-CUB-FP32-'.gpu.'-'.dist.'.dat'
ompDouble   = 'data/alg-omp-double-'.cpu.'-'.dist.'-B32.dat'

plot    singlepass  using 1:($1/($5*1000000)) title "Single-pass"     with lp ls 1,\
        cub16       using 1:($1/($2*1000000)) title "CUB (half)"      with lp ls 3,\
        cub32       using 1:($1/($2*1000000)) title "CUB (float)"     with lp ls 2,\
        ompDouble   using 1:($1/($5*1000000)) title "OpenMP (nt=10)" with lp ls 4
