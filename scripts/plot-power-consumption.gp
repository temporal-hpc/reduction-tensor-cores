reset
if (ARGC != 3){
    print "run as : gnuplot -c  CPU-MODEL GPU-MODEL DISTRIBUTION"
    exit
}

cpu  = ARG1
gpu  = ARG2
dist = ARG3

print "plot-power.gp\n CPU = ",cpu,"\n GPU = ",gpu,"\n dist: ",dist
out = 'plots/power-consumption-'.cpu.'-'.gpu.'-'.dist.'.eps'
mytitle = "Power Consumption (".gpu.")\nn = 400M, repeats = 1000"

set autoscale                        # scale axes automatically
set term postscript eps color blacktext "Courier" 18
set output out
set title mytitle font "Courier, 22"

set ytics mirror
set ylabel 'W' rotate by 0 offset 1
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set xrange [0:46]
set log x

set xlabel 'time [s]'
set font "Courier, 20"
set pointsize   0.5
#set xtics format "%1.0s"
set xtics (0.1, 1, 2, 5, 10, 20, 42)
set key right top Left  font "Courier, 18"

set style line 1 lt 1 lc rgb 'forest-green' dt 1    pt 5    pi -6   lw 3 # green   
set style line 2 lt 2 lc rgb 'black'        dt "--"        pt 2    pi -6   lw 3 # orange
set style line 3 lt 3 lc rgb 'web-blue'     dt "."      pt 6    pi -6   lw 3 # blue
set style line 4 lt 4 lc rgb 'red'          dt 5        pt 11   pi -6   lw 3 # purple

# variables
single_pass_data    = 'data/power-singlepass-'.gpu.'.dat'
cub16_data          = 'data/power-CUB-half-'.gpu.'.dat'
cub32_data          = 'data/power-CUB-float-'.gpu.'.dat'
omp_data            = 'data/power-omp-nt10-double-'.cpu.'.dat'

#print "warp_shuffle_data: ".warp_shuffle_data
#print "split_data: ".split_data
#print "recurrence_data: ".recurrence_data
#print "single_pass_data: ".single_pass_data

plot\
        cub16_data          using 6:2 title "CUB (half)"       with l   ls 3,\
        cub32_data          using 6:2 title "CUB (float)"      with l   ls 2,\
        omp_data            using 6:2 title "OpenMP (nt=10)"   with l   ls 4,\
        single_pass_data    using 6:2 title "Single-pass"      with l   ls 1

print "done!\n\n"
