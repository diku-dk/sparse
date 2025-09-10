
FUTHARK?=futhark

.PHONY: all
all: test

.PHONY: clean
clean:
	rm -rf *~ *.c compressed_test.fut.* mono_test.fut.* doc mono compressed trapezoidal_test.fut.* triangular_test.fut.* blocked_square_regular_test.fut.* trapezoidal trapezoidal_test triangular triangular_test compressed_test mono_test blocked_square_regular blocked_square_regular_test

.PHONY: futversion
futversion:
	$(FUTHARK) --version

.PHONY: test
test:
	$(FUTHARK) test --pass-compiler-option=--Werror compressed_test.fut mono_test.fut trapezoidal_test.fut triangular_test.fut blocked_square_regular_test.fut

.PHONY: doc
doc:
	$(FUTHARK) doc -o doc .
