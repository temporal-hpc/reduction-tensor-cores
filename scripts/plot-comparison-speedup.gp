reset

gpu  = ARG1
dist = ARG2
cpu  = ARG3

print "plot-comparison-speedup.gp ---> GPU: ",gpu," dist: ",dist

out     = 'plots/comparison-speedup-'.gpu.'-'.dist.'.eps'
mytitle = "Speedup vs CUB Library (".gpu.")\n".dist." Distribution\n"

set autoscale # scale axes automatically
set term postscript eps color blacktext "Courier" 24
set output out
set title mytitle

set ylabel 'Speedup' rotate by 90 offset 0
#set yrange [1:4.5]

set xlabel 'N x 10^{6}'
set font "Courier, 20"
set pointsize   0.5
set xtics format "%1.0s"
set key Left bot right reverse samplen 3.0 font "Courier,18" spacing 1 

set style line 1 lt 1 lc rgb 'forest-green' dt 1    pt 5    pi -6   lw 2 # green   
set style line 2 lt 2 lc rgb 'black'       dt 2    pt 2    pi -6   lw 2 # orange
set style line 3 lt 3 lc rgb 'web-blue'     dt 6    pt 6    pi -6   lw 2 # blue
set style line 4 lt 4 lc rgb 'red'          dt 5    pt 11   pi -6   lw 2 # purple
set style line 5 lt 1 lc rgb '#77ac30'              pt 13   pi -6   lw 2 # green
set style line 6 lt 1 lc rgb '#4dbeee'              pt 4    pi -6   lw 2 # light-blue
set style line 7 lt 1 lc rgb '#a2142f'              pt 8    pi -6   lw 2 # red

# variables
singlepass  = '< paste data/alg-warpshuffle-'.gpu.'-'.dist.'-B1024.dat data/alg-singlepass-'.gpu.'-'.dist.'-B128.dat'
cub16       = '< paste data/alg-warpshuffle-'.gpu.'-'.dist.'-B1024.dat data/alg-CUB-FP16-'.gpu.'-'.dist.'.dat'
cub32       = '< paste data/alg-warpshuffle-'.gpu.'-'.dist.'-B1024.dat data/alg-CUB-FP32-'.gpu.'-'.dist.'.dat'
ompFloat    = '< paste data/alg-warpshuffle-'.gpu.'-'.dist.'-B1024.dat data/alg-omp-float-'.cpu.'-'.dist.'-B1.dat'
ompDouble   = '< paste data/alg-warpshuffle-'.gpu.'-'.dist.'-B1024.dat data/alg-omp-double-'.cpu.'-'.dist.'-B1.dat'
  

plot    singlepass  using 1:($5/$17) title "single-pass" with lp ls 1,\
        cub16       using 1:($5/$14) title "CUB (half)"  with lp ls 3,\
        cub32       using 1:($5/$14) title "CUB (float)" with lp ls 2,\
        ompFloat    using 1:($5/$17) title "OpenMP ".cpu." (float)" with lp ls 4,\
        ompDouble   using 1:($5/$17) title "OpenMP ".cpu." (double)" with lp ls 5
