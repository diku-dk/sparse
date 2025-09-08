-- | ignore

import "blocked_square_regular"
import "../linalg/linalg"
import "../linalg/lu"

module mat = blocked_square_regular f64 { def bsz=2i64 }
module linalg = mk_linalg f64
module lu = mk_lu f64

def eqv [n] (x:[n]f64) (y:[n]f64) =
  map2 (f64.==) x y |> reduce (&&) true

def eq [n] (a:[n][n]f64) (b:[n][n]f64) =
  map2 eqv a b |> reduce (&&) true

def lower [n] 'a (z:a) (one:a)  (x:[n][n]a) : [n][n]a =
  tabulate_2d n n (\r c -> if r==c then one else if r > c then x[r][c] else z)

def upper [n] 'a (z:a) (x:[n][n]a) : [n][n]a =
  tabulate_2d n n (\r c -> if r <= c then x[r][c] else z)

def z = 0.0f64
def one = 1.0f64

open mat

def matmul = linalg.matmul

-- ==
-- entry: test_simple
-- input { 0i64 }
-- output { true }

entry test_simple (_e:i64) : bool =
--  let b = lu_dense [[1,2],[3,4]]
--  let b' = matmul (lower z one b) (upper z b)
  let a = mk 4 [(0,0,[[1,2],[3,4]]:>[bsz][bsz]t),
		(1,1,[[1,2],[3,4]]:>[bsz][bsz]t)]
  let x = dense (blu_nofill a)
  let x' = matmul (lower z one x) (upper z x)
  in eq (dense a) x' && dim a == 4

-- ==
-- entry: test_eye
-- input { 4i64 } output { true }
-- input { 8i64 } output { true }
-- input { 0i64 } output { true }

entry test_eye (n:i64) : bool =
  let expected = tabulate_2d n n (\r c -> if r == c then 1.0 else 0.0)
  in eq (dense(eye n)) expected && dim (eye n) == n

-- ==
-- entry: test_diag
-- input { empty([0]f64) } output { true }
-- input { [1f64,2f64,3f64,4f64] } output { true }
-- input { [1f64,2f64,3f64,4f64,5f64,6f64,7f64,8f64] } output { true }

entry test_diag [n] (a:[n]f64) : bool =
  let expected = tabulate_2d n n (\r c -> if r == c then a[r] else 0.0)
  in eq (dense(diag a)) expected && dim (diag a) == n

-- ==
-- entry: test_transp
-- input { empty([0]f64) } output { true }
-- input { [1f64,2f64,3f64,4f64] } output { true }
-- input { [1f64,2f64,3f64,4f64,5f64,6f64,7f64,8f64] } output { true }

entry test_transp [n] (a:[n]f64) : bool =
  let expected_diag = tabulate_2d n n (\r c -> if r == c then a[r] else 0.0)
  let b = mk 4 [(0,0,[[1,2],[3,4]]:>[bsz][bsz]t),
		(1,1,[[1,2],[3,4]]:>[bsz][bsz]t)]
  let expected_b : [4][4]f64 = [[1,2,0,0],
				[3,4,0,0],
				[0,0,1,2],
				[0,0,3,4]]
  in eq (dense(transp(diag a))) expected_diag
     && dim (transp(diag a)) == n
     && eq (dense(transp b)) (transpose(dense b))
     && eq (transpose(dense b)) (transpose expected_b)
     && eq (dense b) expected_b

def b = mk 4 [(0,0,[[1,2],[3,4]]:>[bsz][bsz]t),
	      (1,1,[[1,2],[3,4]]:>[bsz][bsz]t)]

def mk_diag_blks (nb:i64) : [nb](i64,i64,[bsz][bsz]t) =
  tabulate nb (\i ->
		 (i,i,tabulate_2d bsz bsz (\r c -> f64.i64(c+1+2*i + r*(c*(r % 2))-r))))

def mk_diag_blk (n:i64) : mat[n] =
  assert (n % bsz == 0)
	 (mk n (mk_diag_blks (n / bsz)))

type op = #ADD | #SUB | #MUL
def test_op (op:op) (n:i64) : bool =
  let opr (a: mat[n]) (b: mat[n]) : mat[n] =
    match op
    case #ADD -> add a b
    case #SUB -> add b (add b (sub a b))
    case #MUL -> mul a b
  in assert (n > 0 && n % bsz == 0)
	    (let nb = n / bsz
             let c = mk n [(nb-1,0,tabulate_2d bsz bsz (\r c -> f64.i64(bsz*r+c+1)))]
	     in eq (dense (opr c (mk_diag_blk n))) (dense (opr (mk_diag_blk n) c)) &&
		eq (dense (opr (transp c) (transp (mk_diag_blk n))))
		   (dense (transp (opr c (mk_diag_blk n)))))

-- ==
-- entry: test_add
-- input { 2i64 } output { true }
-- input { 4i64 } output { true }
-- input { 8i64 } output { true }
entry test_add (n:i64) : bool =
  test_op #ADD n

-- ==
-- entry: test_mul
-- input { 2i64 } output { true }
-- input { 4i64 } output { true }
-- input { 8i64 } output { true }
entry test_mul (n:i64) : bool =
  test_op #MUL n

-- ==
-- entry: test_sub
-- input { 2i64 } output { true }
-- input { 4i64 } output { true }
-- input { 8i64 } output { true }
entry test_sub (n:i64) : bool =
  test_op #SUB n

-- ==
-- entry: test_smsmm
-- input { 2i64 } output { true }
-- input { 4i64 } output { true }
-- input { 8i64 } output { true }
entry test_smsmm (n:i64) : bool =
  let nb = n / bsz
  let c = mk n [(nb-1,0,tabulate_2d bsz bsz (\r c -> f64.i64(bsz*r+c+1)))]
  let v = map (+2) (iota n) with [0] = 10
  let k = diag (map f64.i64 v)
  in eq (matmul (dense c) (dense k)) (dense(smsmm c k))

-- ==
-- entry: test_smvm
-- input { 2i64 } output { true }
-- input { 4i64 } output { true }
-- input { 8i64 } output { true }
entry test_smvm (n:i64) : bool =
  let nb = n / bsz
  let c = mk n [(nb-1,0,tabulate_2d bsz bsz (\r c -> f64.i64(bsz*r+c+1)))]
  let v = map (+2) (iota n) with [0] = 10
  let v = map f64.i64 v
  in eqv (linalg.matvecmul_row (dense c) v) (smvm c v)

-- -- input { 8i64 } output { true }

def g2 () : mat [2] =
  let b00 = tabulate_2d bsz bsz (\i j -> f64.i64(i + j + 1)) :> [bsz][bsz]f64
  in mk 2 [(0,0,b00)]

def g4 () : mat [4] =
  let b00 = [[1,-2],[3,-9]] :> [bsz][bsz]f64
  let b01 = [[-2,-3],[0,-9]] :> [bsz][bsz]f64
  let b10 = [[-1,2],[-3,-6]] :> [bsz][bsz]f64
  let b11 = [[4,7],[26,2]] :> [bsz][bsz]f64
  in mk 4 [ (0,0,b00), (0,1,b01), (1,0,b10), (1,1,b11)]

def g6() : mat [6] =
  let d = tabulate_2d 6 6 (\i j -> if (i < 4 && i > 1 && j < 2) || (i < 4 && j > 3)
				   then f64.i64 0
				   else f64.sqrt(f64.i64(i*3+j+1)))
  let b00 = d[:2,:2] :> [bsz][bsz]f64
  let b01 = d[:2,2:4] :> [bsz][bsz]f64
  let b11 = d[2:4,2:4] :> [bsz][bsz]f64
  let b20 = d[4:6,:2] :> [bsz][bsz]f64
  let b21 = d[4:6,2:4] :> [bsz][bsz]f64
  let b22 = d[4:6,4:6] :> [bsz][bsz]f64
  in mk 6 [(0,0,b00),(0,1,b01),(1,1,b11),(2,0,b20),(2,1,b21),(2,2,b22)]

-- ==
-- entry: test_blu_nofill
-- input { 2i64 } output { true }
-- input { 4i64 } output { true }
-- input { 6i64 } output { true }
entry test_blu_nofill (n:i64) : bool =
  let m : mat [n] = (if n == 4 then g4()
		     else if n == 6 then g6()
		     else g2()) :> mat [n]
  in eq (dense (blu_nofill m)) (lu.lu 2 (dense m))
