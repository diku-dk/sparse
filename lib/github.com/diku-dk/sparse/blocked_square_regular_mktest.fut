-- | ignore

import "blocked_square_regular"
import "../linalg/linalg"
import "../linalg/lup"
import "../linalg/perm"

module linalg = mk_linalg f64
module lup64 = mk_lup f64

def matmul = linalg.matmul

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

local module mk_test ( X : { val bsz : i64 } ) = {

  module mat = blocked_square_regular f64 { def bsz = X.bsz }

  open mat

  def test_eye (nb:i64) : bool =
    let n = nb * bsz
    let expected = tabulate_2d n n (\r c -> if r == c then 1.0 else 0.0)
    in eq (dense(eye n)) expected && dim (eye n) == n

  def test_diag (nb:i64) : bool =
    let n = nb*bsz
    let a = iota n |> map (\i -> f64.i64 (i + 1))
    let expected = tabulate_2d n n (\r c -> if r == c then a[r] else 0.0)
    in eq (dense(diag a)) expected && dim (diag a) == n

  def test_transp (nb:i64) : bool =
    let n = nb*bsz
    let a = iota n |> map (\i -> f64.i64 (i + 1))
    let expected_diag = tabulate_2d n n (\r c -> if r == c then a[r] else 0.0)
    let bl = tabulate_2d bsz bsz (\i j -> f64.i64(i*bsz+j+1))
    let bz = tabulate_2d bsz bsz (\_ _ -> 0)
    let b = mk (2*bsz) [(0, 0, bl),(1, 1, bl)]
    let expected_b = linalg.block bl bz bz bl :> [2*bsz][2*bsz]f64
    in eq (dense(transp(diag a))) expected_diag
       && dim (transp(diag a)) == n
       && eq (dense(transp b)) (transpose(dense b))
       && eq (transpose(dense b)) (transpose expected_b)
       && eq (dense b) expected_b

  def mk_diag_blks (nb:i64) : [nb](i64,i64,[bsz][bsz]t) =
    tabulate nb (\i ->
                   (i,i,tabulate_2d bsz bsz (\r c -> f64.i64(c+1+2*i + r*(c*(r % 2))-r))))

  def mk_diag_blk (nb:i64) : mat[nb*bsz] =
    mk (nb*bsz) (mk_diag_blks nb)

  type op = #ADD | #SUB | #MUL
  def test_op (op:op) (nb:i64) : bool =
    let opr (a: mat[nb*bsz]) (b: mat[nb*bsz]) : mat[nb*bsz] =
      match op
      case #ADD -> add a b
      case #SUB -> add b (add b (sub a b))
      case #MUL -> mul a b
    in assert (nb > 0)
              (let c = mk (nb*bsz) [(nb-1,0,tabulate_2d bsz bsz (\r c -> f64.i64(bsz*r+c+1)))]
               in eq (dense (opr c (mk_diag_blk nb))) (dense (opr (mk_diag_blk nb) c)) &&
                  eq (dense (opr (transp c) (transp (mk_diag_blk nb))))
                     (dense (transp (opr c (mk_diag_blk nb)))))

  def test_add (nb:i64) : bool =
    test_op #ADD nb

  def test_mul (nb:i64) : bool =
    test_op #MUL nb

  def test_sub (nb:i64) : bool =
    test_op #SUB nb

  def test_smsmm (nb:i64) : bool =
    let c = mk (nb*bsz) [(nb-1,0,tabulate_2d bsz bsz (\r c -> f64.i64(bsz*r+c+1)))]
    let v = map (+2) (iota (nb*bsz)) with [0] = 10
    let k = diag (map f64.i64 v)
    in eq (matmul (dense c) (dense k)) (dense(smsmm c k))

  def test_smvm (nb:i64) : bool =
    let c = mk (nb*bsz) [(nb-1,0,tabulate_2d bsz bsz (\r c -> f64.i64(bsz*r+c+1)))]
    let v = map (+2) (iota (nb*bsz)) with [0] = 10
    let v = map f64.i64 v
    in eqv (linalg.matvecmul_row (dense c) v) (smvm c v)

  def mk_blkdiag (nb:i64) : mat[nb*bsz] =
    let f i a = f64.sqrt (f64.i64 (i+1)) + (f64.i64 (a+i) |> f64.sin |> (*28.0))
    let diag_blks = map (\i -> (i,i,unflatten (map (f i) (iota (bsz*bsz))))) (iota nb)
    in mk (nb*bsz) diag_blks

  def test_solve_sparse (nb:i64) : bool =
    let m = mk_blkdiag nb
    let blk = tabulate_2d bsz bsz (\i j -> f64.sin(f64.sqrt(f64.i64(2*j*bsz+3*i+2))))
    let m = if nb >= 5 then add m (mk (nb*bsz) [(3,4,blk)])
            else m
    let m_dense = copy(dense m)
    let (lu,p) = lup m
    let L = lower lu
    let U = upper lu
    let LU = smsmm L U
    let LU_dense = dense LU
    in eq_eps 0.00006 (perm.permute p m_dense) LU_dense

  def test_solve_sparse2 (nb:i64) : bool =
    let m1 = mk_blkdiag nb
    let m2 = transp (mk_blkdiag nb)
    let m = add m1 m2
    let blk = tabulate_2d bsz bsz (\i j -> f64.sin(f64.sqrt(f64.i64(2*j*bsz+3*i+2))))
    let m = if nb >= 5 then add m (mk (nb*bsz) [(3,4,blk)])
            else m
    let m_dense = copy(dense m)
    let (lu,p) = lup m
    let L = lower lu
    let U = upper lu
    let LU = smsmm L U
    let LU_dense = dense LU
    in eq_eps 0.00006 (perm.permute p m_dense) LU_dense

  def test_ols (nb:i64) : bool =
    let m1 = mk_blkdiag nb
    let m2 = transp (mk_blkdiag nb)
    let m = add m1 m2
    let blk = tabulate_2d bsz bsz (\i j -> f64.sin(f64.sqrt(f64.i64(2*j*bsz+3*i+2))))
    let m = if nb >= 5 then add m (mk (nb*bsz) [(3,4,blk)])
            else m
    let b = map (\i -> f64.i64(i+2)) (iota (nb*bsz))
    let x = ols m b
    let b' = smvm m x
    in eqv_eps 0.00006 b b'

  def test_solve_sparse2nopiv (nb:i64) : bool =
    let m1 = mk_blkdiag nb
    let m2 = transp (mk_blkdiag nb)
    let m = add m1 m2
    let m_dense = copy(dense m)
    let lu' = lu m
    let L = lower lu'
    let U = upper lu'
    let LU = smsmm L U
    let LU_dense = dense LU
    in eq_eps 0.00006 m_dense LU_dense

  def all f xs = map f xs |> reduce (&&) true

  def test_all : bool =
    all test_solve_sparse2nopiv (3...8)
    && all test_ols (3...8)
    && all test_solve_sparse2 (3...8)
    && all test_solve_sparse (1...8)
    && all test_smvm (1...8)
    && all test_smsmm (1...8)
    && all test_mul (1...8)
    && all test_sub (1...8)
    && all test_add (1...8)
    && all test_transp (0...4)
    && all test_diag (0...4)
    && all test_eye (0...4)
}

local module t2 = mk_test { def bsz = 2i64 }
entry test2 : bool = t2.test_all
-- ==
-- entry: test2
-- input { } output { true }

local module t3 = mk_test { def bsz = 3i64 }
entry test3 : bool = t3.test_all
-- ==
-- entry: test3
-- input { } output { true }

local module t4 = mk_test { def bsz = 4i64 }
entry test4 : bool = t4.test_all
-- ==
-- entry: test4
-- input { } output { true }
