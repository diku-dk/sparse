-- | Mono sparse matrices.
--
-- A mono sparse matrix is a matrix that stores fewer elements than a
-- corresponding dense regular matrix (non-stored elements are assumed
-- to be zero). There are two kinds of mono sparse matrices, one that
-- stores only one element per row (mono sparse row matrix) and one
-- that stores only one element per column (mono sparse column
-- matrix).

import "../segmented/segmented"
import "../linalg/linalg"
import "../sorts/merge_sort"

import "matrix_regular"

-- | Module type including submodules for mono sparse row matrix
-- operations (`sr`) and mono sparse column matrix operations
-- (`sc`). Notice that the abstract matrix types `sr` and `sc` are
-- *not* size-lifted as their representations are regular.  The module
-- type is declared `local` to avoid that outside code makes direct
-- use of the module type (allowing it to be extended in minor
-- revisions).

local module type mono = {
  type t
  type sr [n][m]
  type sc [n][m]

  -- | Mono sparse row representation.
  module sr : {
    include matrix_regular with t = t
                           with mat [n][m] = sr[n][m]
    -- | Matrix transposition.
    val transpose [n][m] : mat[n][m] -> sc[m][n]
    -- | Sparse matrix vector multiplication. Given a sparse `n` times
    -- `m` matrix and a vector of size `m`, the function returns a
    -- vector of size `n`, the result of multiplying the argument
    -- matrix with the argument vector.
    val smvm      [n][m] : mat[n][m] -> [m]t -> [n]t
    -- | Vector sparse matrix multiplication.
    val vsmm      [n][m] : [n]t -> mat[n][m] -> [m]t
    -- | Dense matrix sparse matrix multiplication.
    val dmsmm  [n][m][k] : [n][k]t -> mat[k][m] -> [n][m]t
  }

  -- | Mono sparse column representation.
  module sc : {
    include matrix_regular with t = t
                           with mat [n][m] = sc [n][m]
    -- | Matrix transposition.
    val transpose [n][m] : mat[n][m] -> sr[m][n]
    -- | Vector sparse matrix multiplication.
    val vsmm      [n][m] : [n]t -> mat[n][m] -> [m]t
  }

}

-- | Parameterised mono sparse matrix module with submodules for the
-- mono sparse row (MSR) representation and for the mono sparse column
-- (MSR) representation. The module is parameterised over a field
-- (defined in the linalg package).

module mk_mono (T : field) : mono with t = T.t = {

  type t = T.t

  -- sorting and merging of coo values
  type coo [nnz] = [nnz](i64,i64,t)

  def zero_val = T.i64 0
  def one_val = T.i64 1
  def eq a b = !(a T.< b) && !(b T.< a)

  def sort_coo [nnz] (coo: coo[nnz]) : coo[nnz] =
    merge_sort (\(r1,c1,_) (r2,c2,_) -> r1 < r2 || (r1 == r2 && c1 <= c2))
               coo

  def merge_coo [nnz] (coo: coo[nnz]) : coo[] =
    let flags = map2 (\(r1,c1,_) (r2,c2,_) -> r1!=r2 || c1!=c2)
                     coo
                     (rotate (-1) coo)
    let flags = if nnz >= 1 then flags with [0] = true
		else flags

    in segmented_reduce (\(r1,c1,v1) (_,_,v2) -> (r1,c1,v1 T.+ v2))
                        (0,0,zero_val) flags coo

  def norm_coo [nnz] (coo: coo[nnz]) : coo[] =
    sort_coo coo |> merge_coo


  -- mono sparse row
  module sr = {
    type t = t
    type mat[n][m] = {col_idx:[n]i64, vals: [n]t, dummy_m: [m]()}

    def zero (n:i64) (m:i64) : mat[n][m] =
      {col_idx=replicate n 0,
       vals=replicate n zero_val,
       dummy_m=replicate m ()}

    def eye (n:i64) (m:i64) : mat[n][m] =
      {col_idx=iota n,
       vals=replicate n one_val,
       dummy_m=replicate m ()}

    def diag [n] (v:[n]t) : mat[n][n] =
      {col_idx=iota n,
       vals=v,
       dummy_m=replicate n ()}

    def dense [n][m] ({col_idx,vals,dummy_m=_}: mat[n][m]) : [n][m]t =
      let A = tabulate_2d n m (\_ _ -> zero_val)
      in scatter_2d A (zip (iota n) col_idx) vals

    def scale [n][m] (v:t) ({col_idx,vals,dummy_m}:mat[n][m]) : mat[n][m] =
      {col_idx, vals=map (T.* v) vals, dummy_m}

    def sparse [nnz0] (n:i64) (m:i64) (coo:coo[nnz0]) : mat[n][m] =
      let [nnz] coo : coo[nnz] = norm_coo coo
      let _ = map (\(r,c,_) -> assert (0 <= r && r < n && 0 <= c && c < m) 0) coo
      let () = if nnz > 1
	       then let _ = map2 (\(r1,_,_) (r2,_,_) -> assert (r1!=r2) ()) coo (rotate 1 coo)
		    in ()
	       else ()
      let (rs,cs,vs) = unzip3 coo
      let vals = scatter (replicate n zero_val) rs vs
      let col_idx = scatter (replicate n 0) rs cs
      in {col_idx, vals, dummy_m=replicate m ()}

    def nnz [n][m] (a: mat[n][m]) : i64 =
      map (\v -> if eq v zero_val then 0 else 1) a.vals
      |> reduce (+) 0

    def coo [n][m] ({col_idx,vals,dummy_m=_}: mat[n][m]) : ?[nnz].[nnz](i64,i64,t) =
      zip3 (iota n) col_idx vals
      |> filter (\(_,_,v) -> v T.< zero_val || zero_val T.< v)

    def (+) [n][m] ({col_idx,vals,dummy_m}: mat[n][m])
                   ({col_idx=col_idx',vals=vals',dummy_m=_}: mat[n][m]) : mat[n][m] =
      let _ = map2 (\c c' -> assert (c==c') ()) col_idx col_idx'
      in {dummy_m=dummy_m, col_idx=col_idx,
	  vals=map2 (T.+) vals vals'}

    def (-) [n][m] ({col_idx,vals,dummy_m}: mat[n][m])
                   ({col_idx=col_idx',vals=vals',dummy_m=_}: mat[n][m]) : mat[n][m] =
      let _ = map2 (\c c' -> assert (c==c') ()) col_idx col_idx'
      in {dummy_m=dummy_m, col_idx=col_idx,
	  vals=map2 (T.-) vals vals'}

    def transpose [n][m] (mat:mat[n][m]) : mat[n][m] =
      mat

    def smvm [n][m] ({col_idx,vals,dummy_m=_}:mat[n][m]) (v:[m]t) : [n]t =
      map2 (\c w -> w T.* v[c]) col_idx vals

    def vsmm [n][m] (v:[n]t) ({col_idx,vals,dummy_m=_}:mat[n][m]) : [m]t =
      let (is,xs) = map3 (\x i y -> (i,x T.* y)) v col_idx vals |> unzip
      in reduce_by_index (replicate m (T.i64 0)) (T.+) (T.i64 0) is xs

    def dmsmm [n][m][k] (D:[n][k]t) (S:mat[k][m]) : [n][m]t =
      map (\r -> vsmm r S) D
  }

  -- mono sparse column
  module sc = {

    type t = t

    def zero (n:i64) (m:i64) : sr.mat[m][n] =
      sr.zero m n

    def scale [n][m] (v:t) (mat:sr.mat[n][m]) : sr.mat[n][m] =
      sr.scale v mat

    def eye (n:i64) (m:i64) : sr.mat[m][n] =
      sr.eye m n

    def diag [n] (v:[n]t) : sr.mat[n][n] =
      sr.diag v

    def nnz [n][m] (mat:sr.mat[n][m]) : i64 =
      sr.nnz mat

    def coo [n][m] (mat: sr.mat[n][m]) : ?[nnz].[nnz](i64,i64,t) =
      sr.coo mat |> map (\(r,c,v) -> (c,r,v))

    def sparse [nnz] (n:i64) (m:i64) (coo: [nnz](i64,i64,t)) : sr.mat[m][n] =
      map (\(r,c,v) -> (c,r,v)) coo |> sr.sparse m n

    def dense [n][m] (mat: sr.mat[n][m]) : [m][n]t =
      sr.dense mat |> transpose

    def (+) x y = x sr.+ y
    def (-) x y = x sr.- y

    def transpose [n][m] (mat:sr.mat[n][m]) : sr.mat[n][m] =
      mat

    def vsmm [n][m] (a:[n]t) (b:sr.mat[m][n]) : [m]t =
      sr.smvm (transpose b) a

    type mat[n][m] = sr.mat[m][n]
  }

  type sr[n][m] = sr.mat[n][m]
  type sc[n][m] = sc.mat[n][m]

}
