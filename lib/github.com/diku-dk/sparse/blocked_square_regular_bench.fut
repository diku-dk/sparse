-- | ignore

import "blocked_square_regular"
import "../linalg/linalg"
import "../linalg/lup"

module mat = blocked_square_regular f64 { def bsz=64i64 }

module linalg = mk_linalg f64
module lup64 = mk_lup f64

local open mat

type coo = (i64,i64,t)

def band (f:i64->i64->f64) (n:i64) (k:i64) : []coo =
  iota (n - i64.abs k) |> map (\i -> if k >= 0 then (i,i+k,f i (i+k))
				     else (i+i64.abs k,i,f (i+i64.abs k) i))

def triband (n:i64) : []coo =
  let f i j : f64 = 10000 * f64.sin (f64.i64(n*i+j+1))
  in band f n 0 ++ band f n 1 ++ band (\i j -> f64.sqrt (f64.abs(f i j))) n (-1)

def pentaband (n:i64) : []coo =
  let f s i j : f64 = 10000 * f64.sin (f64.i64(n*i+j+1+s))
  let g s i j : f64 = f64.sqrt (f64.abs(f s i j))
  in band (f 0) n 0 ++ band (f 1) n 1 ++ band (g 0) n (-1)
	  ++ band (f 2) n 2 ++ band (f 3) n (-2)

def ok (x:f64) : bool =
  !(f64.isnan x || f64.isinf x)

def eqv [n] (eps:f64) (a:[n]f64) (b:[n]f64) : bool =
  map2 (\x y -> assert(ok x && ok y)(f64.abs(x-y) < eps)) a b |> reduce (&&) true

def coo2dense [k] (n:i64) (xs:[k]coo) : [n][n]f64 =
  let m = replicate n (replicate n 0.0)
  in scatter_2d m (map (\(i,j,_) -> (i,j)) xs) (map (.2) xs)

def run_sparse [n] (A:mat[n]) : bool =
  let b = iota n |> map f64.i64
  let x = ols A b
  let b' = smvm A x
  in eqv 0.05 b b'

def run_dense [n] (A:[n][n]f64) : bool =
  let b = iota n |> map f64.i64
  let x = lup64.ols (copy A) (copy b)
  let b' = linalg.matvecmul_row A x
  in eqv 0.05 b b'

import "../linalg/lu"
module lu64 = mk_lu f64

def run_blkdense [n] (A:[n][n]f64) : bool =
  let b = iota n |> map f64.i64
  let x = lu64.ols 32 (copy A) (copy b)
  let b' = linalg.matvecmul_row A x
  in eqv 0.05 b b'

entry main_sparse_triband (n:i64) : bool =
  run_sparse (triband n |> from_coo n)

entry main_dense_triband (n:i64) : bool =
  run_dense (triband n |> coo2dense n)

entry main_blkdense_triband (n:i64) : bool =
  run_blkdense (triband n |> coo2dense n)

entry main_sparse_pentaband (n:i64) : bool =
  run_sparse (pentaband n |> from_coo n)

entry main_dense_pentaband (n:i64) : bool =
  run_dense (pentaband n |> coo2dense n)

entry main_blkdense_pentaband (n:i64) : bool =
  run_blkdense (pentaband n |> coo2dense n)


-- ==
-- entry: main_sparse_pentaband
-- input { 64i64 } output { true }
-- input { 128i64 } output { true }
-- input { 256i64 } output { true }
-- input { 512i64 } output { true }
-- input { 1024i64 } output { true }
-- input { 2048i64 } output { true }
-- input { 2944i64 } output { true }
-- input { 4096i64 } output { true }
-- input { 5952i64 } output { true }
-- input { 8192i64 } output { true }

-- ==
-- entry: main_dense_pentaband
-- input { 64i64 } output { true }
-- input { 128i64 } output { true }
-- input { 256i64 } output { true }
-- input { 512i64 } output { true }
-- input { 1024i64 } output { true }
-- input { 2048i64 } output { true }
-- input { 2944i64 } output { true }

-- ==
-- entry: main_blkdense_pentaband
-- input { 64i64 } output { true }
-- input { 128i64 } output { true }
-- input { 256i64 } output { true }
-- input { 512i64 } output { true }
-- input { 1024i64 } output { true }
-- input { 2048i64 } output { true }
-- input { 2944i64 } output { true }
-- input { 4096i64 } output { true }
-- input { 5952i64 } output { true }
-- input { 8192i64 } output { true }

-- ==
-- entry: main_sparse_triband
-- input { 64i64 } output { true }
-- input { 128i64 } output { true }
-- input { 256i64 } output { true }
-- input { 512i64 } output { true }
-- input { 1024i64 } output { true }
-- input { 2048i64 } output { true }
-- input { 2944i64 } output { true }
-- input { 4096i64 } output { true }
-- input { 5952i64 } output { true }
-- input { 8192i64 } output { true }

-- ==
-- entry: main_dense_triband
-- input { 64i64 } output { true }
-- input { 128i64 } output { true }
-- input { 256i64 } output { true }
-- input { 512i64 } output { true }
-- input { 1024i64 } output { true }
-- input { 2048i64 } output { true }
-- input { 2944i64 } output { true }

-- ==
-- entry: main_blkdense_triband
-- input { 64i64 } output { true }
-- input { 128i64 } output { true }
-- input { 256i64 } output { true }
-- input { 512i64 } output { true }
-- input { 1024i64 } output { true }
-- input { 2048i64 } output { true }
-- input { 2944i64 } output { true }
-- input { 4096i64 } output { true }
-- input { 5952i64 } output { true }
-- input { 8192i64 } output { true }
