
FUTHARK?=futhark

.PHONY: all
all: test

.PHONY: clean
clean:
	rm -rf *~ *.c sparse_test sparse_test.fut.*

.PHONY: futversion
futversion:
	$(FUTHARK) --version

.PHONY: test
test:
	$(FUTHARK) test sparse_test.fut
