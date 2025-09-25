-- | ignore

import "blocked_square_regular"
import "../linalg/linalg"
import "../linalg/lup"
import "../linalg/perm"

module mat = blocked_square_regular f64 { def bsz=2i64 }
module linalg = mk_linalg f64
module lup64 = mk_lup f64

def eqv [n] (x:[n]f64) (y:[n]f64) =
  map2 (f64.==) x y |> reduce (&&) true

def eqv_eps [n] (eps:f64) (x:[n]f64) (y:[n]f64) =
  map2 (\a b -> f64.abs(a-b) < eps) x y |> reduce (&&) true

def eqvi [n] (x:[n]i64) (y:[n]i64) =
  map2 (i64.==) x y |> reduce (&&) true

def eq [n] (a:[n][n]f64) (b:[n][n]f64) =
  map2 eqv a b |> reduce (&&) true

def eq_eps [n] (eps:f64) (a:[n][n]f64) (b:[n][n]f64) =
  map2 (eqv_eps eps) a b |> reduce (&&) true

def lower_dense [n] 'a (z:a) (one:a)  (x:[n][n]a) : [n][n]a =
  tabulate_2d n n (\r c -> if r==c then one else if r > c then x[r][c] else z)

def upper_dense [n] 'a (z:a) (x:[n][n]a) : [n][n]a =
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
--  let b' = matmul (lower_dense z one b) (upper_dense z b)
  let a = mk 4 [(0,0,[[1,2],[3,4]]:>[bsz][bsz]t),
                (1,1,[[1,2],[3,4]]:>[bsz][bsz]t)]
  let (lu,p) = lup_nofill a
  let x = dense lu
  let x' = matmul (lower_dense z one x) (upper_dense z x)
  in eq (perm.permute p (dense (copy a))) x' && dim a == 4

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

entry test_solve_full : [4]t =
  let a00 = [[3.0,-7],[-3.0,5]] :> [bsz][bsz]t
  let a01 = [[-2.0,2],[1.0,0]] :> [bsz][bsz]t
  let a10 = [[6.0,-4],[-9.0,5]] :> [bsz][bsz]t
  let a11 = [[0.0,-5],[-5.0,12]] :> [bsz][bsz]t
  let a = mk 4 [ (0,0,a00), (0,1,a01), (1,0,a10), (1,1,a11)]
  let b = [-9.0,5,7,11]
  let (lu,p) = lup a
  let dlu = dense lu
  let pb = perm.permute p b
  let y = lup64.forsolve dlu pb
  let x = lup64.backsolve dlu y
  in x

-- ==
-- entry: test_solve_full
-- input { } output { [3.0,4,-6,-1] }

entry test_solve_dense4 : [4]t =
  let a = [[3.0,-7,-2,2],
           [-3.0,5,1,0],
           [6.0,-4,0,-5],
           [-9.0,5,-5,12]]
  let b = [-9.0,5,7,11]
  let (lu,p) = lup64.lup a
  let pb = perm.permute p b
  let y = lup64.forsolve lu pb
  let x = lup64.backsolve lu y
  in x

-- ==
-- entry: test_solve_dense4
-- input { } output { [3.0,4,-6,-1] }

def g14 () : mat [14] =
  let d = diag (map f64.i64 (iota 14))
  let b = tabulate_2d bsz bsz (\r c -> f64.i64(r*bsz+c+1))
  let u = mk 14 [(0,2,b),(0,5,b),(1,3,b),(0,6,b)]
  let l = mk 14 [(3,1,b),(4,0,b),(5,1,b)]
  in add d (add u l)

def mk_blkdiag (n:i64) : mat[n*bsz] =
  let f i a = f64.sqrt (f64.i64 (i+1)) + (f64.i64 (a+i) |> f64.sin |> (*28.0))
  let diag_blks = map (\i -> (i,i,unflatten (map (f i) (iota (bsz*bsz))))) (iota n)
  in mk (n*bsz) diag_blks

def solve_sparse (n:i64) : [n*bsz][n*bsz]t =
  let m = mk_blkdiag n
  let (lu,_p) = lup (copy m)
  let L = lower lu
  in dense L
--  let U = upper lu
--   let LU = smsmm L U
--   let LU_dense = dense LU
--   let m_dense = dense m
--   in eq (perm.permute p m_dense) LU_dense

entry test_solve_sparse (n:i64) : bool =
  let m = mk_blkdiag n
  let blk = [[3.0,2],[7.0,-1]] :> [bsz][bsz]t
  let m = if n >= 5 then add m (mk (n*bsz) [(3,4,blk)])
          else m
  let m_dense = copy(dense m)
  let (lu,p) = lup m
  let L = lower lu
  let U = upper lu
  let LU = smsmm L U
  let LU_dense = dense LU
  in eq_eps 0.00006 (perm.permute p m_dense) LU_dense

-- ==
-- entry: test_solve_sparse
-- input { 3i64 } output { true }
-- input { 4i64 } output { true }
-- input { 5i64 } output { true }
-- input { 8i64 } output { true }

entry test_solve_sparse2 (n:i64) : bool =
  let m1 = mk_blkdiag n
  let m2 = transp (mk_blkdiag n)
  let m = add m1 m2
  let blk = [[3.0,2],[7.0,-1]] :> [bsz][bsz]t
  let m = if n >= 5 then add m (mk (n*bsz) [(3,4,blk)])
          else m
  let m_dense = copy(dense m)
  let (lu,p) = lup m
  let L = lower lu
  let U = upper lu
  let LU = smsmm L U
  let LU_dense = dense LU
  in eq_eps 0.00006 (perm.permute p m_dense) LU_dense

-- ==
-- entry: test_solve_sparse2
-- input { 3i64 } output { true }
-- input { 4i64 } output { true }
-- input { 5i64 } output { true }
-- input { 8i64 } output { true }

entry test_ols (n:i64) : bool =
  let m1 = mk_blkdiag n
  let m2 = transp (mk_blkdiag n)
  let m = add m1 m2
--  let blk = [[3.0,2],[7.0,-1]] :> [bsz][bsz]t
--  let m = if n >= 5 then add m (mk (n*bsz) [(3,4,blk)])
--          else m
  let b = map (\i -> f64.i64(i+2)) (iota (n*bsz))
  let x = ols m b
  let b' = smvm m x
  in eqv_eps 0.00006 b b'

-- ==
-- entry: test_ols
-- input { 3i64 } output { true }
-- input { 4i64 } output { true }
-- input { 5i64 } output { true }
-- input { 8i64 } output { true }

entry test_solve_sparse2nopiv (n:i64) : bool =
  let m1 = mk_blkdiag n
  let m2 = transp (mk_blkdiag n)
  let m = add m1 m2
  let m_dense = copy(dense m)
  let lu' = lu m
  let L = lower lu'
  let U = upper lu'
  let LU = smsmm L U
  let LU_dense = dense LU
  in eq_eps 0.00006 m_dense LU_dense

-- ==
-- entry: test_solve_sparse2nopiv
-- input { 3i64 } output { true }
-- input { 4i64 } output { true }
-- input { 5i64 } output { true }
-- input { 8i64 } output { true }

entry test_forsolve : [4]f64 =
  let m00 = [[1.0,0],[2.0,1]] :> [bsz][bsz]t
  let m10 = [[3.0,4],[-1.0,-3]] :> [bsz][bsz]t
  let m11 = [[1.0,0],[0.0,1]] :> [bsz][bsz]t
  let m = mk 4 [(0,0,m00),(1,0,m10),(1,1,m11)]
  let b = [8.0,7,14,-7]
  let y = forsolve m b
  in y

-- entry: test_forsolve
-- input { } output { [8.0,-9,26,-26] }

entry test_backsolve : [4]f64 =
  let m00 = [[1.0,1],[0.0,-1]] :> [bsz][bsz]t
  let m01 = [[0.0,3],[-1.0,-5]] :> [bsz][bsz]t
  let m11 = [[3.0,13],[0.0,-13]] :> [bsz][bsz]t
  let m = mk 4 [(0,0,m00),(0,1,m01),(1,1,m11)]
  let y = [8.0,-9,26,-26]
  let x = backsolve m y
  in x

-- entry: test_backsolve
-- input { } output { [3.0,-1,0,2] }

entry test_lu_find_fills (_:i64) : ([]i64,[]i64) =
  let a = g14()
  in lu_find_fills a |> unzip

-- ==
-- entry: test_lu_find_fills
-- input { 2i64 } output { [4i64,4i64,4i64,5i64] [2i64,5i64,6i64,3i64] }

entry test_from_coo (n:i64) (xs:[]i64) (ys:[]i64) (vs:[]t) : [n][n]t =
  zip3 xs ys vs |> from_coo n |> dense

-- ==
-- entry: test_from_coo
-- input { 4i64 [2i64] [1i64] [3.0] }
-- output { [[0f64,0,0,0],[0f64,0,0,0],[0f64,3.0,0,0],[0f64,0,0,0]] }
-- input { 4i64 empty([0]i64) empty([0]i64) empty([0]f64) }
-- output { [[0f64,0,0,0],[0f64,0,0,0],[0f64,0.0,0,0],[0f64,0,0,0]] }
-- input { 2i64 [0i64] [1i64] [3.0] } output { [[0f64,3],[0f64,0]] }
-- input { 2i64 [0i64,1] [1i64,0] [3.0,2.0] } output { [[0f64,3],[2f64,0]] }
-- input { 4i64 [2i64,0] [1i64,2] [3.0,1.0] } output { [[0f64,0,1,0],[0f64,0,0,0],[0f64,3.0,0,0],[0f64,0,0,0]] }

entry test_coo [k] (n:i64) (xs:[k]i64) (ys:[k]i64) (vs:[k]t) : bool =
  let m = zip3 xs ys vs |> from_coo n
  let cs = coo m :> [k](i64,i64,t)
  in eqvi xs (map (.0) cs) && eqvi ys (map (.1) cs) && eqv vs (map (.2) cs)

-- ==
-- entry: test_coo
-- input { 4i64 [0i64,2] [3i64,1] [-2.0,3.0] } output { true }
