
import "sparse"

module sp = csr { open i32 def fma a b c : i32 = a * b + c }

-- ==
-- entry: test_eye
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

entry test_eye (n:i64) (m:i64) : *[n][m]i32 =
  sp.eye n m |> sp.dense

-- ==
-- entry: test_sparse
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [[2,0,0],[0,0,3]] }
-- input { 2i64 3i64 empty([0]i64) empty([0]i64) empty([0]i32) }
-- output { [[0,0,0],[0,0,0]] }

entry test_sparse [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]sp.t) : *[n][m]i32 =
  sp.sparse n m (zip3 xs ys vs) |> sp.dense

-- ==
-- entry: test_smvm
-- input { 5i64 5i64
--         [0i64,0i64,0i64,1i64,1i64,2i64,2i64,2i64,3i64,4i64,4i64]
--         [0i64,1i64,3i64,1i64,2i64,1i64,2i64,3i64,3i64,3i64,4i64]
--         [1,2,11,3,4,5,6,7,8,9,10]
--         [3,1,2,6,5]
-- }
-- output { [71,11,59,48,104] }

entry test_smvm [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]sp.t) (v:*[m]sp.t) : *[n]sp.t =
  let m = sp.sparse n m (zip3 xs ys vs)
  in sp.smvm m v

-- ==
-- entry: test_nnz
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { 2i64 }
-- input { 2i64 3i64 empty([0]i64) empty([0]i64) empty([0]i32) }
-- output { 0i64 }

entry test_nnz [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]sp.t) : i64 =
  sp.sparse n m (zip3 xs ys vs) |> sp.nnz

-- ==
-- entry: test_coo
-- input { 2i64 3i64 [0i64,1i64] [0i64,2i64] [2,3] }
-- output { [0i64,1i64] [0i64,2i64] [2,3] }

entry test_coo [k] (n:i64) (m:i64) (xs:[k]i64) (ys:[k]i64) (vs: [k]sp.t)
    : ([]i64,[]i64,[]sp.t) =
  sp.sparse n m (zip3 xs ys vs) |> sp.coo |> unzip3
