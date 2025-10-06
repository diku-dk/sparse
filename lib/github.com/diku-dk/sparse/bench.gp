DATAFILE="bench.csv"
M=1000000
set grid
set datafile separator ","
set terminal png font "sans-serif,10"
set key left autotitle columnheader
set ylabel 'seconds'
set xlabel 'N (dimension of NxN matrix)'

#set xdata n

set output "bench.png"
set title "Pentaband Multicore LU-Decomposition"
plot DATAFILE using 1:($2/M) index 1 with linespoints, \
           '' using 1:($2/M) index 2 with linespoints, \
	   '' using 1:($2/M) index 3 with linespoints

set output "bench-tri.png"
set title "Triband Multicore LU-Decomposition"
plot DATAFILE using 1:($2/M) index 4 with linespoints, \
           '' using 1:($2/M) index 5 with linespoints, \
	   '' using 1:($2/M) index 6 with linespoints