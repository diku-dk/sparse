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
-- operations (`msr`) and mono sparse column matrix operations
-- (`msc`). Notice that the abstract matrix types `msr` and `msc` are
-- *not* size-lifted as their representations are regular.  The module
-- type is declared `local` to avoid that outside code makes direct
-- use of the module type (allowing it to be extended in minor
-- revisions).

local module type mono = {
  type t
  type msr [n][m]
  type msc [n][m]

  -- | Mono sparse row
  module msr : {
    include matrix_regular with t = t
                           with mat [n][m] = msr[n][m]
    -- | Matrix transposition.
    val transpose [n][m] : mat[n][m] -> msc[m][n]
    -- | Sparse matrix vector multiplication. Given a sparse `n` times
    -- `m` matrix and a vector of size `m`, the function returns a
    -- vector of size `n`, the result of multiplying the argument
    -- matrix with the argument vector.
    val smvm      [n][m] : mat[n][m] -> [m]t -> [n]t
  }

  -- | Mono sparse column
  module msc : {
    include matrix_regular with t = t
                           with mat [n][m] = msc [n][m]
    -- | Matrix transposition.
    val transpose [n][m] : mat[n][m] -> msr[m][n]
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
  module msr = {
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

  }

  -- mono sparse column
  module msc = {

    type t = t

    def zero (n:i64) (m:i64) : msr.mat[m][n] =
      msr.zero m n

    def scale [n][m] (v:t) (mat:msr.mat[n][m]) : msr.mat[n][m] =
      msr.scale v mat

    def eye (n:i64) (m:i64) : msr.mat[m][n] =
      msr.eye m n

    def nnz [n][m] (mat:msr.mat[n][m]) : i64 =
      msr.nnz mat

    def coo [n][m] (mat: msr.mat[n][m]) : ?[nnz].[nnz](i64,i64,t) =
      msr.coo mat |> map (\(r,c,v) -> (c,r,v))

    def sparse [nnz] (n:i64) (m:i64) (coo: [nnz](i64,i64,t)) : msr.mat[m][n] =
      map (\(r,c,v) -> (c,r,v)) coo |> msr.sparse m n

    def dense [n][m] (mat: msr.mat[n][m]) : [m][n]t =
      msr.dense mat |> transpose

    def (+) x y = x msr.+ y
    def (-) x y = x msr.- y

    def transpose [n][m] (mat:msr.mat[n][m]) : msr.mat[n][m] =
      mat

    type mat[n][m] = msr.mat[m][n]
  }

  type msr[n][m] = msr.mat[n][m]
  type msc[n][m] = msc.mat[n][m]

}
