
import "sparse"

module sparse = mk_sparse { open i32 def fma a b c : i32 = a * b + c }

-- *************
-- MSR Tests
-- *************

module msr = sparse.msr

-- ==
-- entry: test_msr_eye
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

entry test_msr_eye (n:i64) (m:i64) : *[n][m]i32 =
  msr.eye n m |> msr.dense

-- ==
-- entry: test_msr_sparse
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [[2,0,0],[0,0,3]] }
-- input { 2i64 3i64 [1i64,0i64] [2i64,0i64] [3,2] }
-- output { [[2,0,0],[0,0,3]] }
-- input { 2i64 3i64 empty([0]i64) empty([0]i64) empty([0]i32) }
-- output { [[0,0,0],[0,0,0]] }

entry test_msr_sparse [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]msr.t) : *[n][m]i32 =
  msr.sparse n m (zip3 xs ys vs) |> msr.dense

-- ==
-- entry: test_msr_smvm
-- input { 5i64 5i64
--         [0i64,1i64,2i64,3i64,4i64]
--         [0i64,1i64,1i64,4i64,3i64]
--         [1,3,8,6,9]
--         [3,10,2,6,5]
-- }
-- output { [3,30,80,30,54] }

entry test_msr_smvm [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]msr.t) (v:*[m]msr.t) : *[n]msr.t =
  let m = msr.sparse n m (zip3 xs ys vs)
  in msr.smvm m v

-- ==
-- entry: test_msr_nnz
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { 2i64 }
-- input { 2i64 3i64 empty([0]i64) empty([0]i64) empty([0]i32) }
-- output { 0i64 }

entry test_msr_nnz [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]msr.t) : i64 =
  msr.sparse n m (zip3 xs ys vs) |> msr.nnz

-- ==
-- entry: test_msr_coo
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [0i64,1i64] [0i64,2i64] [2,3] }

entry test_msr_coo [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]msr.t)
    : ([]i64,[]i64,[]msr.t) =
  msr.sparse n m (zip3 xs ys vs) |> msr.coo |> unzip3


-- *************
-- MSC Tests
-- *************

module msc = sparse.msc

-- ==
-- entry: test_msc_eye
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

entry test_msc_eye (n:i64) (m:i64) : *[n][m]i32 =
  msc.eye n m |> msc.dense

-- ==
-- entry: test_msc_sparse
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [[2,0,0],[0,0,3]] }
-- input { 2i64 3i64 [1i64,0i64] [2i64,0i64] [3,2] }
-- output { [[2,0,0],[0,0,3]] }
-- input { 2i64 3i64 empty([0]i64) empty([0]i64) empty([0]i32) }
-- output { [[0,0,0],[0,0,0]] }

entry test_msc_sparse [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]msc.t) : *[n][m]i32 =
  msc.sparse n m (zip3 xs ys vs) |> msc.dense

-- ==
-- entry: test_msc_nnz
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { 2i64 }
-- input { 2i64 3i64 empty([0]i64) empty([0]i64) empty([0]i32) }
-- output { 0i64 }

entry test_msc_nnz [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]msc.t) : i64 =
  msc.sparse n m (zip3 xs ys vs) |> msc.nnz

-- ==
-- entry: test_msc_coo
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [0i64,1i64] [0i64,2i64] [2,3] }

entry test_msc_coo [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]msc.t)
    : ([]i64,[]i64,[]msc.t) =
  msc.sparse n m (zip3 xs ys vs) |> msc.coo |> unzip3

-- ==
-- entry: test_msr_transpose
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [[2,0],[0,0],[0,3]] }

entry test_msr_transpose [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]msr.t)
    : [][]msc.t =
  msr.sparse n m (zip3 xs ys vs) |> msr.transpose |> msc.dense

-- ==
-- entry: test_msc_transpose
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [[2,0],[0,0],[0,3]] }

entry test_msc_transpose [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]msc.t)
    : [][]msr.t =
  msc.sparse n m (zip3 xs ys vs) |> msc.transpose |> msr.dense
