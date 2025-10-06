
FUTHARK?=futhark

.PHONY: all
all: test

.PHONY: clean
clean:
	rm -rf *~ *.c *.fut.* doc mono compressed trapezoidal trapezoidal_test triangular triangular_test compressed_test mono_test blocked_square_regular blocked_square_regular_test blocked_square_regular_mktest blocked_square_regular_bench bench.png bench.csv bench.png bench-tri.png matrix_regular matrix_irregular

.PHONY: futversion
futversion:
	$(FUTHARK) --version

.PHONY: test
test:
	$(FUTHARK) test --pass-compiler-option=--Werror compressed_test.fut mono_test.fut trapezoidal_test.fut triangular_test.fut blocked_square_regular_test.fut blocked_square_regular_mktest.fut

.PHONY: doc
doc:
	$(FUTHARK) doc -o doc .

BENCH_BACKEND?=multicore

.PHONY: bench
bench: bench.tm
	$(MAKE) bench.png

bench.tm: blocked_square_regular_bench.fut
	$(FUTHARK) bench --backend=$(BENCH_BACKEND) $< | tee $@

bench.csv: bench.tm
	cat $< | sed 's/#[0-9]* (\"//g' | sed 's/μs (95%.*//g' | sed 's/i64\"):/, /g' | sed 's/[^:]*:\(.*\)/\n\1/g' | sed 's/ (.*)://g' | sed 's/main_//g' | sed 's/_[a-z]*//g' > $@

bench.png: bench.csv
	gnuplot bench.gp
