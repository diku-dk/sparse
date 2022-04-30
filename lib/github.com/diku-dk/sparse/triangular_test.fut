-- | ignore

import "triangular"

module sparse = mk_triangular { open i32 def fma a b c : i32 = a * b + c }

module upper = sparse.upper

-- ==
-- entry: test_upper_eye
-- input { 2i64 }
-- output { [[1,0],[0,1]] }
-- input { 4i64 }
-- output { [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] }
-- input { 0i64 }
-- output { empty([0][0]i32) }

entry test_upper_eye (n:i64) : *[n][n]i32 =
  upper.eye n |> upper.dense

-- ==
-- entry: test_upper_nnz
-- input { [[1,2,3],[0,4,5],[0,0,6]] }
-- output { 6i64 }
-- input { empty([0][0]i32) }
-- output { 0i64 }
-- input { [[0,0,0],[0,0,0],[0,0,0]] }
-- output { 0i64 }

entry test_upper_nnz [n] (d:[n][n]i32) : i64 =
  upper.triangular d |> upper.nnz

-- ==
-- entry: test_upper_triangular
-- input { [[1,2,3],[4,5,6],[7,8,9]] }
-- output { [[1,2,3],[0,5,6],[0,0,9]] }

entry test_upper_triangular [n] (a:[n][n]i32) : [n][n]i32 =
  upper.triangular a |> upper.dense

module lower = sparse.lower

-- ==
-- entry: test_lower_eye
-- input { 2i64 }
-- output { [[1,0],[0,1]] }
-- input { 4i64 }
-- output { [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] }
-- input { 0i64 }
-- output { empty([0][0]i32) }

entry test_lower_eye (n:i64) : *[n][n]i32 =
  lower.eye n |> lower.dense

-- ==
-- entry: test_lower_triangular
-- input { [[1,2,3],[4,5,6],[7,8,9]] }
-- output { [[1,0,0],[4,5,0],[7,8,9]] }

entry test_lower_triangular [n] (a:[n][n]i32) : [n][n]i32 =
  lower.triangular a |> lower.dense

-- *******************
-- TRANSPOSE TESTS
-- *******************

-- ==
-- entry: test_upper_transpose
-- input { [[1,2,3],[0,4,5],[0,0,6]] }
-- output { [[1,0,0],[2,4,0],[3,5,6]] }
-- input { empty([0][0]i32) }
-- output { empty([0][0]i32) }

entry test_upper_transpose [n] (a:[n][n]i32) : [n][n]i32 =
  upper.triangular a |> upper.transpose |> lower.dense

-- ==
-- entry: test_lower_transpose
-- input { [[1,0,0],[2,4,0],[3,5,6]] }
-- output { [[1,2,3],[0,4,5],[0,0,6]] }
-- input { empty([0][0]i32) }
-- output { empty([0][0]i32) }

entry test_lower_transpose [n] (a:[n][n]i32) : [n][n]i32 =
  lower.triangular a |> lower.transpose |> upper.dense

module m = mk_triangular f64

-- ==
-- entry: test_transpose
-- input { [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]] }
-- output { [[1.0, 0.0, 0.0], [2.0, 4.0, 0.0], [3.0, 5.0, 6.0]] }
entry test_transpose [n] (A: [n][n]f64) =
  m.lower.dense (m.upper.transpose (m.upper.triangular A))
