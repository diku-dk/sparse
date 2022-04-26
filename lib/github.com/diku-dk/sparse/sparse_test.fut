
import "sparse"

module sparse = mk_sparse { open i32 def fma a b c : i32 = a * b + c }

-- *************
-- CSR Tests
-- *************

module csr = sparse.csr

-- ==
-- entry: test_csr_eye
-- input { 2i64 2i64 }
-- output { [[1,0],[0,1]] }
-- input { 2i64 3i64 }
-- output { [[1,0,0],[0,1,0]] }
-- input { 3i64 2i64 }
-- output { [[1,0],[0,1],[0,0]] }
-- input { 1i64 3i64 }
-- output { [[1,0,0]] }
-- input { 0i64 0i64 }
-- output { empty([0][0]i32) }
-- input { 0i64 1i64 }
-- output { empty([0][1]i32) }
-- input { 1i64 0i64 }
-- output { empty([1][0]i32) }

entry test_csr_eye (n:i64) (m:i64) : *[n][m]i32 =
  csr.eye n m |> csr.dense

-- ==
-- entry: test_csr_sparse
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [[2,0,0],[0,0,3]] }
-- input { 2i64 3i64 [1i64,0i64] [2i64,0i64] [3,2] }
-- output { [[2,0,0],[0,0,3]] }
-- input { 2i64 3i64 [1i64,0i64,1i64] [2i64,0i64,2i64] [3,2,1] }
-- output { [[2,0,0],[0,0,4]] }
-- input { 2i64 3i64 empty([0]i64) empty([0]i64) empty([0]i32) }
-- output { [[0,0,0],[0,0,0]] }

entry test_csr_sparse [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]csr.t) : *[n][m]i32 =
  csr.sparse n m (zip3 xs ys vs) |> csr.dense

-- ==
-- entry: test_csr_smvm
-- input { 5i64 5i64
--         [0i64,0i64,0i64,1i64,1i64,2i64,2i64,2i64,3i64,4i64,4i64]
--         [0i64,1i64,3i64,1i64,2i64,1i64,2i64,3i64,3i64,3i64,4i64]
--         [1,2,11,3,4,5,6,7,8,9,10]
--         [3,1,2,6,5]
-- }
-- output { [71,11,59,48,104] }

entry test_csr_smvm [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]csr.t) (v:*[m]csr.t) : *[n]csr.t =
  let m = csr.sparse n m (zip3 xs ys vs)
  in csr.smvm m v

-- ==
-- entry: test_csr_nnz
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { 2i64 }
-- input { 2i64 3i64 empty([0]i64) empty([0]i64) empty([0]i32) }
-- output { 0i64 }

entry test_csr_nnz [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]csr.t) : i64 =
  csr.sparse n m (zip3 xs ys vs) |> csr.nnz

-- ==
-- entry: test_csr_coo
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [0i64,1i64] [0i64,2i64] [2,3] }

entry test_csr_coo [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]csr.t)
    : ([]i64,[]i64,[]csr.t) =
  csr.sparse n m (zip3 xs ys vs) |> csr.coo |> unzip3


-- *************
-- CSC Tests
-- *************

module csc = sparse.csc

-- ==
-- entry: test_csc_eye
-- input { 2i64 2i64 }
-- output { [[1,0],[0,1]] }
-- input { 2i64 3i64 }
-- output { [[1,0,0],[0,1,0]] }
-- input { 3i64 2i64 }
-- output { [[1,0],[0,1],[0,0]] }
-- input { 1i64 3i64 }
-- output { [[1,0,0]] }
-- input { 0i64 0i64 }
-- output { empty([0][0]i32) }
-- input { 0i64 1i64 }
-- output { empty([0][1]i32) }
-- input { 1i64 0i64 }
-- output { empty([1][0]i32) }

entry test_csc_eye (n:i64) (m:i64) : *[n][m]i32 =
  csc.eye n m |> csc.dense

-- ==
-- entry: test_csc_sparse
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [[2,0,0],[0,0,3]] }
-- input { 2i64 3i64 [1i64,0i64] [2i64,0i64] [3,2] }
-- output { [[2,0,0],[0,0,3]] }
-- input { 2i64 3i64 [1i64,0i64,1i64] [2i64,0i64,2i64] [3,2,1] }
-- output { [[2,0,0],[0,0,4]] }
-- input { 2i64 3i64 empty([0]i64) empty([0]i64) empty([0]i32) }
-- output { [[0,0,0],[0,0,0]] }

entry test_csc_sparse [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]csc.t) : *[n][m]i32 =
  csc.sparse n m (zip3 xs ys vs) |> csc.dense

-- ==
-- entry: test_csc_nnz
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { 2i64 }
-- input { 2i64 3i64 empty([0]i64) empty([0]i64) empty([0]i32) }
-- output { 0i64 }

entry test_csc_nnz [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]csc.t) : i64 =
  csc.sparse n m (zip3 xs ys vs) |> csc.nnz

-- ==
-- entry: test_csc_coo
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [0i64,1i64] [0i64,2i64] [2,3] }

entry test_csc_coo [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]csc.t)
    : ([]i64,[]i64,[]csc.t) =
  csc.sparse n m (zip3 xs ys vs) |> csc.coo |> unzip3

-- ==
-- entry: test_csr_transpose
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [[2,0],[0,0],[0,3]] }

entry test_csr_transpose [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]csr.t)
    : [][]csc.t =
  csr.sparse n m (zip3 xs ys vs) |> csr.transpose |> csc.dense

-- ==
-- entry: test_csc_transpose
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [[2,0],[0,0],[0,3]] }

entry test_csc_transpose [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]csc.t)
    : [][]csr.t =
  csc.sparse n m (zip3 xs ys vs) |> csc.transpose |> csr.dense

-- ==
-- entry: test_smm
-- input { 2i64 2i64 2i64 [0i64] [1i64] [1] [1i64] [0i64] [1] }
-- output { [[1,0],[0,0]] }
-- input { 2i64 2i64 2i64 [1i64] [0i64] [1] [0i64] [1i64] [1] }
-- output { [[0,0],[0,1]] }
-- input { 2i64 3i64 4i64 [1i64] [0i64] [5] empty([0]i64) empty([0]i64) empty([0]i32) }
-- output { [[0,0,0,0],[0,0,0,0]] }
-- input { 2i64 2i64 2i64 [0i64,1i64] [0i64,1i64] [1,1] [0i64,1i64] [1i64,0i64] [8,9] }
-- output { [[0,8],[9,0]] }
-- input { 2i64 2i64 2i64 [0i64,0i64,1i64,1i64] [0i64,1i64,0i64,1i64] [1,7,2,4]
--                        [0i64,0i64,1i64,1i64] [0i64,1i64,0i64,1i64] [3,3,5,2] }
-- output { [[38,17],[26,14]] }

entry test_smm [k1][k2] (n:i64) (m:i64) (k:i64)
                        (xs1:[k1]i64) (ys1:[k1]i64) (vs1: [k1]csc.t)
                        (xs2:[k2]i64) (ys2:[k2]i64) (vs2: [k2]csc.t)
    : [][]csr.t =
  let A = csr.sparse n m (zip3 xs1 ys1 vs1)
  let B = csc.sparse m k (zip3 xs2 ys2 vs2)
  in sparse.smm A B |> csr.dense
