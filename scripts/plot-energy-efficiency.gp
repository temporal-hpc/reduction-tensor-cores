reset
if (ARGC != 3){
    print "run as : gnuplot -c  GPU-MODEL  CPU-MODEL DISTRIBUTION"
    exit
}
cpu  = ARG1
gpu  = ARG2
dist = ARG3

print "plot-energy-efficiency.gp\n CPU = ",cpu,"\n GPU = ",gpu,"\n dist: ",dist
out = 'plots/energy-efficiency-'.cpu.'-'.gpu.'-'.dist.'.eps'
mytitle = "Energy Efficiency (".gpu.")\nn = 400M, repeats = 1000"

set autoscale                        # scale axes automatically
set term postscript eps color blacktext "Courier" 20
set output out
set title mytitle

set ylabel "Total Energy (J)" rotate by 90 offset 2 font "Courier, 18"
set y2label "BEPS/W" rotate by 90 offset -0.5 font "Courier, 18"
set xrange [-0.5:5.5]
set y2range [0:2.5]
set log y
set boxwidth 0.5

set xlabel font "Courier, 15"
set xtics ("Single-pass" 0.25, "CUB-half" 1.75, "CUB-float" 3.25, "OpenMP-nt10" 4.75)
set xtics font "Courier, 17"
set ytics font "Courier, 16"
set ytics nomirror  (10, 100, 210, 360, 1000, 3000, 10000)
set y2tics nomirror (0.5, 1.0, 1.5, 2.0, 2.5)
set y2tics font "Courier, 16"
set key right top Left  font "Courier, 16"

set style fill solid

# variables
data    = 'data/energy-efficiency-'.gpu.'.dat'

plot\
    data every 2 u 1:2 axes x1y1 with boxes title "Total Energy" ls 3,\
    data every 2 u 1:2:2 axes x1y1 with labels offset char 0,0.5 font "Courier, 10" title "",\
    data every 2::1 u 1:5  axes x1y2 with boxes title "BEPS per Watt" ls 2,\
    data every 2::1 u 1:5:5 axes x1y2 with labels offset char 0,0.5 font "Courier, 10" title ""

print "done!\n\n"
