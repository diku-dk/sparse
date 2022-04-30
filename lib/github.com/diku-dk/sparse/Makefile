
FUTHARK?=futhark

.PHONY: all
all: test

.PHONY: clean
clean:
	rm -rf *~ *.c sparse_test sparse_test.fut.* sparse_test_msr_msc sparse_test_msr_msc.fut.* doc sparse trapezoidal_test.fut.* triangular_test.fut.* trapezoidal trapezoidal_test triangular triangular_test

.PHONY: futversion
futversion:
	$(FUTHARK) --version

.PHONY: test
test:
	$(FUTHARK) test --pass-compiler-option=--Werror sparse_test.fut sparse_test_msr_msc.fut trapezoidal_test.fut triangular_test.fut
