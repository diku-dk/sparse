-- | ignore

import "triangular"

module m = mk_triangular f64

-- ==
-- entry: test_transpose
-- input { [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]] }
-- output { [[1.0, 0.0, 0.0], [2.0, 4.0, 0.0], [3.0, 5.0, 6.0]] }
entry test_transpose [n] (A: [n][n]f64) =
  m.lower.dense (m.upper.transpose (m.upper.triangular A))
