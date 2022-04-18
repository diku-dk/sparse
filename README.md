# Futhark sparse matrix operations [![CI](https://github.com/diku-dk/sparse/workflows/CI/badge.svg)](https://github.com/diku-dk/sparse/actions) [![Documentation](https://futhark-lang.org/pkgs/github.com/diku-dk/sparse/status.svg)](https://futhark-lang.org/pkgs/github.com/diku-dk/sparse/latest/)

A library for sparse matrix operations in Futhark.

## Installation

```
$ futhark pkg add github.com/diku-dk/sparse
$ futhark pkg sync
```

## Usage

```
> import "lib/github.com/diku-dk/sparse/sparse"
> module sp = sparse f64
> let A = sp.csr.sparse 2 3 [(0,0,2),(1,2,3)]
> sp.csr.smvm A [10,20,30]
[20,90]
```

## See also

* https://github.com/diku-dk/segmented
* https://github.com/diku-dk/linalg
