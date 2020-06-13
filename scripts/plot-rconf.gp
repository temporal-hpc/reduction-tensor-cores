reset
gpu  = ARG1
dist = ARG2
alg = ARG3
y1 = ARG4
y2 = ARG5

print "plot-rconf.gp ---> GPU: ",gpu," dist: ",dist," alg: ",alg
out = 'plots/rconf-'.alg.'-'.gpu.'-'.dist.'.eps'
title = "Running time (".gpu.")\n".alg.", N {/Symbol \273} 100M" 

# scale axes automatically
set autoscale
set term postscript eps color blacktext "Courier" 24
set output out
set title title

set yrange [y1:y2]
set ytics mirror
set xtics (0, 4, 8, 16, 24, 32, 40, 48, 56, 64)
set ylabel 'Time [ms]' rotate by 90 offset -1
set xlabel '#R'
set font "Courier, 20"

set key Left top left reverse samplen 3.0 font "Courier,18" spacing 1 
set style line 1 lt 1 lc rgb 'forest-green' dt 1 pt 5   pi -2 lw 2 ps 1 # green   
set style line 2 lt 2 lc rgb 'orange' dt 2 pt 7         pi -2 lw 2 ps 1 # orange
set style line 3 lt 3 lc rgb 'web-blue' dt 6 pt 6       pi -2 lw 2 ps 1 # blue
set style line 4 lt 4 lc rgb 'red' dt 5 pt 11           pi -1 lw 2 ps 1 # purple
set style line 5 lt 1 lc rgb '#77ac30' pt 13            pi -2 lw 2 ps 1 # green
set style line 6 lt 1 lc rgb '#4dbeee' pt 4             pi -2 lw 2 ps 1 # light-blue
set style line 7 lt 1 lc rgb '#a2142f' pt 8             pi -2 lw 2 ps 1 # red

set key right top Left font "Courier, 20"

plot 'data/rconf-'.alg.'-'.gpu.'-'.dist.'-B32.dat' using 3:4 title "B32" with lp ls 1,\
     'data/rconf-'.alg.'-'.gpu.'-'.dist.'-B128.dat' using 3:4 title "B128" with lp ls 4,\
     'data/rconf-'.alg.'-'.gpu.'-'.dist.'-B512.dat' using 3:4 title "B512" with lp ls 3,\
     'data/rconf-'.alg.'-'.gpu.'-'.dist.'-B1024.dat' using 3:4 title "B1024" with lp ls 2
