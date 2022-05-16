# Futhark sparse matrix operations [![CI](https://github.com/diku-dk/sparse/workflows/CI/badge.svg)](https://github.com/diku-dk/sparse/actions) [![Documentation](https://futhark-lang.org/pkgs/github.com/diku-dk/sparse/status.svg)](https://futhark-lang.org/pkgs/github.com/diku-dk/sparse/latest/)

A library for sparse matrix operations in Futhark.

## Installation

```
$ futhark pkg add github.com/diku-dk/sparse
$ futhark pkg sync
```

## Usage

```
$ futhark repl
[0]> import "lib/github.com/diku-dk/sparse/compressed"
[1]> module compressed = mk_compressed f64
[2]> let A = compressed.sr.sparse 2 3 [(0,0,2),(1,2,3)]
[3]> compressed.sr.smvm A [10,20,30]
[20.0f64, 90.0f64]
```

## See also

* https://github.com/diku-dk/segmented
* https://github.com/diku-dk/linalg
