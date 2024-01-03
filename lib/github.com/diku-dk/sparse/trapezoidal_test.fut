-- | ignore

import "trapezoidal"

module sparse = mk_trapezoidal i32

module upper = sparse.upper

-- ==
-- entry: test_upper_eye
-- input { 2i64 }
-- output { [[1,0],[0,1]] }
-- input { 4i64 }
-- output { [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] }

-- -- input { 0i64 }
-- -- output { empty([0][0]i32) }

entry test_upper_eye (n:i64) : *[n][n]i32 =
  upper.eye n n |> upper.dense

-- ==
-- entry: test_upper_nnz
-- input { [[1,2,3],[0,4,5],[0,0,6]] }
-- output { 6i64 }
-- input { [[0,0,0],[0,0,0],[0,0,0]] }
-- output { 0i64 }

-- -- input { empty([0][0]i32) }
-- -- output { 0i64 }

entry test_upper_nnz [n] (d:[n][n]i32) : i64 =
  upper.trapezoidal d |> upper.nnz

-- ==
-- entry: test_upper_trapezoidal
-- input { [[1,2,3],[4,5,6],[7,8,9]] }
-- output { [[1,2,3],[0,5,6],[0,0,9]] }

entry test_upper_trapezoidal [n] (a:[n][n]i32) : [n][n]i32 =
  upper.trapezoidal a |> upper.dense

module lower = sparse.lower

-- ==
-- entry: test_lower_eye
-- input { 2i64 }
-- output { [[1,0],[0,1]] }
-- input { 4i64 }
-- output { [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] }

-- -- input { 0i64 }
-- -- output { empty([0][0]i32) }

entry test_lower_eye (n:i64) : *[n][n]i32 =
  lower.eye n n |> lower.dense

-- ==
-- entry: test_lower_trapezoidal
-- input { [[1,2,3],[4,5,6],[7,8,9]] }
-- output { [[1,0,0],[4,5,0],[7,8,9]] }
-- input { [[1,0],[3,4],[5,6],[7,8]] }
-- output { [[1,0],[3,4],[5,6],[7,8]] }
-- input { [[1,0,0],[4,5,0]] }
-- output { [[1,0,0],[4,5,0]] }
entry test_lower_trapezoidal [n][m] (a:[n][m]i32) : [n][m]i32 =
  lower.trapezoidal a |> lower.dense

-- *******************
-- TRANSPOSE TESTS
-- *******************

-- ==
-- entry: test_upper_transpose
-- input { [[1,2,3],[0,4,5],[0,0,6]] }
-- output { [[1,0,0],[2,4,0],[3,5,6]] }

-- -- input { empty([0][0]i32) }
-- -- output { empty([0][0]i32) }

entry test_upper_transpose [n] (a:[n][n]i32) : [n][n]i32 =
  upper.trapezoidal a |> upper.transpose |> lower.dense

-- ==
-- entry: test_lower_transpose
-- input { [[1,0,0],[2,4,0],[3,5,6]] }
-- output { [[1,2,3],[0,4,5],[0,0,6]] }

-- -- input { empty([0][0]i32) }
-- -- output { empty([0][0]i32) }

entry test_lower_transpose [n] (a:[n][n]i32) : [n][n]i32 =
  lower.trapezoidal a |> lower.transpose |> upper.dense

-- ==
-- entry: test_lower_smm
-- input { [[1,0],[0,1]] [[1,0],[0,1]] }
-- output { [[1,0],[0,1]] }
-- input { [[1,0],[3,4]] [[1,0],[3,4]] }
-- output { [[1,0],[15,16]] }
-- input { [[1,0],[3,4],[5,6],[7,8]] [[1,0,0],[4,5,0]] }
-- output { [[1,0,0],[19,20,0],[29,30,0],[39,40,0]] }
-- input { [[1,0,0],[4,5,0],[7,8,9]] [[1,0],[3,4],[5,6]] }
-- output { [[1,0],[19,20],[76,86]] }

entry test_lower_smm [n][m][k] (a:[n][m]i32) (b:[m][k]i32) : [n][k]i32 =
  lower.smm (lower.trapezoidal a) (lower.trapezoidal b) |> lower.dense

-- ==
-- entry: test_upper_smm
-- input { [[1,0],[0,1]] [[1,0],[0,1]] }
-- output { [[1,0],[0,1]] }
-- input { [[1,2],[0,4]] [[10,20],[0,40]] }
-- output { [[10,100],[0,160]] }
entry test_upper_smm [n][m][k] (a:[n][m]i32) (b:[m][k]i32) : [n][k]i32 =
  upper.smm (upper.trapezoidal a) (upper.trapezoidal b) |> upper.dense
