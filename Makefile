FUTHARK?=futhark

.PHONY: test
test:
	$(MAKE) -C lib/github.com/diku-dk/sparse test

.PHONY: clean
clean:
	$(MAKE) -C lib/github.com/diku-dk/sparse clean
	rm -rf *~

.PHONY: sync
sync:
	$(FUTHARK) pkg sync

.PHONY: realclean
realclean:
	$(MAKE) clean
	rm -rf lib/github.com/diku-dk/segmented
	rm -rf lib/github.com/diku-dk/linalg
	rm -rf lib/github.com/diku-dk/cpprandom
